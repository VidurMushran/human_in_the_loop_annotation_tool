# app/training/jobs.py
from __future__ import annotations
import os
import copy
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader

from app.ml.train import train_model, TrainConfig
from app.ml.models import make_model
from app.ml.dataset import Item, H5StreamDataset
from app.ml.score import score_h5_file
from app.data.h5io import read_images_by_indices
from app.experiments.registry import new_run_dir, write_run_yaml, write_metrics, write_checkpoint
from app.utils.model_helpers import smart_load_state_dict

def _create_run_dir(runs_root: str, suffix: str = "") -> Path:
    import time
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{suffix}" if suffix else timestamp
    run_dir = new_run_dir(str(runs_root / dir_name))
    return run_dir

def run_standard_job(
    cfg: TrainConfig, 
    job_name: str, 
    train_items: list, 
    val_items: list, 
    timm_name: str, 
    in_chans: int, 
    runs_root: str, 
    log_cb,
    resume_checkpoint: str = None
):
    log_cb(f"Starting standard job: {job_name}")
    
    start_epoch = 1
    optimizer_state = None
    
    # 1. Initialize Model
    model = make_model(cfg.model_kind, timm_name, in_chans)

    # 2. Resume Logic (Smart Load)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        log_cb(f"Resuming from {resume_checkpoint}...")
        try:
            ckpt = torch.load(resume_checkpoint, map_location="cpu")
            # Use smart load to ignore size mismatches
            smart_load_state_dict(model, ckpt.get("model_state_dict", ckpt))
            
            if "optimizer_state_dict" in ckpt:
                optimizer_state = ckpt["optimizer_state_dict"]
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            log_cb(f"Resumed at epoch {start_epoch}")
        except Exception as e:
            log_cb(f"Resume failed ({e}). Starting fresh.")

    # 3. Setup Directory
    safe_arch = timm_name if cfg.model_kind == "timm_frozen" else "simple_cnn"
    suffix = f"{safe_arch}_{cfg.inputs_mode}_{cfg.training_method}"
    run_dir = _create_run_dir(runs_root, suffix=suffix)

    # 4. Save Callback
    extra = {"job_name": job_name, "mode": cfg.training_method, "inputs": cfg.inputs_mode}
    write_run_yaml(run_dir, cfg=cfg.__dict__, extra=extra)

    def save_callback(epoch, net, opt, stats):
        state = {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
            "train_config": cfg.__dict__,
            "extra": extra,
            "stats": stats
        }
        torch.save(state, run_dir / "checkpoint_latest.pt")

    # 5. Execute Training
    trained_model, best = train_model(
        model, train_items, val_items, cfg, 
        log_cb=log_cb, 
        start_epoch=start_epoch, 
        optimizer_state=optimizer_state, 
        save_cb=save_callback
    )

    # 6. Final Artifacts
    try:
        ckpt_src = run_dir / "checkpoint_src.pt"
        torch.save({
            "model_state_dict": trained_model.state_dict(),
            "train_config": cfg.__dict__, "extra": extra, "best": best
        }, ckpt_src)
        write_checkpoint(run_dir, str(ckpt_src))
        metrics = dict(best) if isinstance(best, dict) else {"best": str(best)}
        metrics.pop("state", None)
        write_metrics(run_dir, metrics)
    except Exception as e:
        log_cb(f"Warning: Artifact save failed: {e}")
        
    return trained_model

def run_pseudo_label_job(cfg: TrainConfig, job_name: str, init_labeled: list, init_unlabeled: list, timm_name: str, in_chans: int, runs_root: str, pl_iters: int, pl_thresh: float, log_cb):
    log_cb(f"Starting Pseudo-Labeling: {job_name}")
    
    current_train = copy.deepcopy(init_labeled)
    current_unlabeled = copy.deepcopy(init_unlabeled)
    
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(current_train))
    n_val = int(0.20 * len(current_train))
    curr_val = [current_train[i] for i in idx[:n_val]]
    curr_train = [current_train[i] for i in idx[n_val:]]
    
    trained_model = None

    for iter_idx in range(pl_iters):
        log_cb(f"\n--- PL Iteration {iter_idx+1}/{pl_iters} ---")
        log_cb(f"Train size: {len(curr_train)} | Unlabeled pool: {len(current_unlabeled)}")
        
        model = make_model(cfg.model_kind, timm_name, in_chans)
        
        cfg.training_method = "supervised"
        trained_model, best = train_model(model, curr_train, curr_val, cfg, log_cb=log_cb)
        
        if iter_idx == pl_iters - 1 or len(current_unlabeled) == 0:
            safe_arch = timm_name if cfg.model_kind == "timm_frozen" else "simple_cnn"
            suffix = f"{safe_arch}_{cfg.inputs_mode}_PL"
            run_dir = _create_run_dir(runs_root, suffix=suffix)
            
            extra = {"job_name": job_name, "mode": "pseudo-labeling", "final_train_size": len(curr_train)}
            write_run_yaml(run_dir, cfg=cfg.__dict__, extra=extra)
            
            ckpt_src = run_dir / "checkpoint_src.pt"
            torch.save({
                "model_state_dict": trained_model.state_dict(),
                "train_config": cfg.__dict__, "best": best
            }, ckpt_src)
            write_checkpoint(run_dir, str(ckpt_src))
            metrics = dict(best) if isinstance(best, dict) else {"best": str(best)}
            write_metrics(run_dir, metrics)
            log_cb(f"Finished. Saved to {run_dir}")
            break
            
        log_cb("Scoring unlabeled pool...")
        trained_model.eval()
        trained_model.to(cfg.device)
        
        files_to_items = defaultdict(list)
        for it in current_unlabeled: files_to_items[it.h5_path].append(it)
        
        new_labeled, remaining = [], []
        
        with torch.no_grad():
            for fp, items in files_to_items.items():
                row_indices = np.array([it.row_idx for it in items])
                for i in range(0, len(items), 256):
                    batch_items = items[i:i+256]
                    batch_idx = row_indices[i:i+256]
                    try:
                        imgs = read_images_by_indices(fp, batch_idx, image_key=cfg.image_key)
                        xb = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
                        if xb.shape[-1] != cfg.target_hw:
                            xb = torch.nn.functional.interpolate(xb, size=(cfg.target_hw, cfg.target_hw))
                        
                        xb = xb.to(cfg.device)
                        
                        # Forward pass through backbone -> classifier
                        # Note: make_model returns a model that outputs embeddings in forward()
                        # We must call model.classifier explicitly or check if it's supervised
                        # In app/ml/train.py we do: emb = model(x); logits = model.classifier(emb)
                        # We must replicate that here:
                        emb = trained_model(xb)
                        logits = trained_model.classifier(emb)
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        
                        for pi, p in enumerate(probs):
                            it = batch_items[pi]
                            if p >= pl_thresh: new_labeled.append(Item(it.h5_path, it.row_idx, 1, it.cluster))
                            elif p <= (1.0-pl_thresh): new_labeled.append(Item(it.h5_path, it.row_idx, 0, it.cluster))
                            else: remaining.append(it)
                    except Exception as e:
                        remaining.extend(batch_items)

        log_cb(f"Pseudo-labeled {len(new_labeled)} samples.")
        curr_train.extend(new_labeled)
        current_unlabeled = remaining

    return trained_model

def run_scoring_job(model, paths, score_col, cfg, log_cb):
    for fp in paths:
        try:
            log_cb(f"Scoring {os.path.basename(fp)} ...")
            score_h5_file(model, fp, score_col=score_col, image_key=cfg.image_key, features_key=cfg.features_key, device="cuda", batch_size=256, target_hw=75, log_cb=log_cb)
        except Exception as e:
            log_cb(f"Error scoring {fp}: {e}")