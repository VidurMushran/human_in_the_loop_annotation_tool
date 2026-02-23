# app/training/jobs.py
from __future__ import annotations
import time
import copy
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from app.ml.train import train_model, TrainConfig
from app.ml.models import SimpleCNN, make_timm_frozen_linear
from app.ml.dataset import Item, H5StreamDataset
from app.ml.score import score_h5_file
from app.experiments.registry import new_run_dir, write_run_yaml, write_metrics, write_checkpoint

def _create_run_dir(runs_root: str, suffix: str = "") -> Path:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{suffix}" if suffix else timestamp
    run_dir = new_run_dir(str(runs_root / dir_name))
    return run_dir

def run_standard_job(cfg: TrainConfig, job_name: str, train_items: list, val_items: list, timm_name: str, in_chans: int, runs_root: str, log_cb):
    log_cb(f"Starting standard job: {job_name}")
    
    if cfg.model_kind == "simple_cnn":
        model = SimpleCNN(in_ch=in_chans, n_classes=2)
    else:
        model = make_timm_frozen_linear(timm_name, in_chans=in_chans, n_classes=2)

    safe_arch = timm_name if cfg.model_kind == "timm_frozen" else "simple_cnn"
    suffix = f"{safe_arch}_{cfg.inputs_mode}_{cfg.training_method}"
    run_dir = _create_run_dir(runs_root, suffix=suffix)

    extra = {
        "job_name": job_name, "mode": cfg.training_method, "inputs": cfg.inputs_mode,
        "training_method": cfg.training_method, "in_chans": in_chans, "n_train": len(train_items)
    }
    write_run_yaml(run_dir, cfg=cfg.__dict__, extra=extra)

    trained_model, best = train_model(model, train_items, val_items, cfg, log_cb=log_cb)

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
    log_cb(f"Starting Pseudo-Labeling job: {job_name}")
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
        
        if cfg.model_kind == "simple_cnn":
            model = SimpleCNN(in_ch=in_chans, n_classes=2)
        else:
            model = make_timm_frozen_linear(timm_name, in_chans=in_chans, n_classes=2)
        
        cfg.training_method = "supervised"
        trained_model, best = train_model(model, curr_train, curr_val, cfg, log_cb=log_cb)
        
        if iter_idx == pl_iters - 1 or len(current_unlabeled) == 0:
            safe_arch = timm_name if cfg.model_kind == "timm_frozen" else "simple_cnn"
            suffix = f"{safe_arch}_{cfg.inputs_mode}_PL"
            run_dir = _create_run_dir(runs_root, suffix=suffix)
            
            extra = {
                "job_name": job_name, "mode": "pseudo-labeling", "inputs": cfg.inputs_mode,
                "training_method": "pseudo-labeling", "in_chans": in_chans, "n_train_final": len(curr_train)
            }
            write_run_yaml(run_dir, cfg=cfg.__dict__, extra=extra)
            
            ckpt_src = run_dir / "checkpoint_src.pt"
            torch.save({
                "model_state_dict": trained_model.state_dict(),
                "train_config": cfg.__dict__, "extra": extra, "best": best
            }, ckpt_src)
            write_checkpoint(run_dir, str(ckpt_src))
            metrics = dict(best) if isinstance(best, dict) else {"best": str(best)}
            write_metrics(run_dir, metrics)
            log_cb(f"Finished. Final run saved to {run_dir}")
            break
            
        log_cb("Scoring unlabeled pool...")
        trained_model.eval()
        trained_model.to(cfg.device)
        
        u_ds = H5StreamDataset(current_unlabeled, image_key=cfg.image_key, target_hw=cfg.target_hw, aug_flags=(), inputs_mode=cfg.inputs_mode, training_method="supervised")
        u_dl = DataLoader(u_ds, batch_size=256, shuffle=False, num_workers=2)
        
        new_labeled, remaining = [], []
        g_idx = 0
        with torch.no_grad():
            for batch in u_dl:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                xb = xb.to(cfg.device)
                probs = torch.softmax(trained_model(xb), dim=1)[:, 1].cpu().numpy()
                for p in probs:
                    it = current_unlabeled[g_idx]
                    if p >= pl_thresh: new_labeled.append(Item(it.h5_path, it.row_idx, 1, it.cluster))
                    elif p <= (1.0 - pl_thresh): new_labeled.append(Item(it.h5_path, it.row_idx, 0, it.cluster))
                    else: remaining.append(it)
                    g_idx += 1
                    
        log_cb(f"Found {len(new_labeled)} high-confidence pseudo-labels.")
        curr_train.extend(new_labeled)
        current_unlabeled = remaining

    return trained_model

def run_scoring_job(model, paths, score_col, cfg, log_cb):
    for fp in paths:
        try:
            log_cb(f"Scoring {os.path.basename(fp)} ...")
            score_h5_file(model, fp, score_col=score_col, image_key=cfg.image_key, features_key=cfg.features_key, device="cuda", batch_size=256, target_hw=75, log_cb=log_cb)
        except Exception as e:
            log_cb(f"[ERROR] Failed to score {os.path.basename(fp)}: {e}")