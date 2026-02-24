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

def execute_job_from_def(job_def, log_cb, progress_cb=None):
    """
    Main entry point for the queue worker to run a job.
    """
    cfg = TrainConfig(**job_def['cfg'])
    
    # 1. Initialize Model
    model = make_model(cfg.model_kind, job_def['timm_name'], job_def['in_chans'])

    # 2. Paths
    runs_root = Path(job_def.get('runs_root', 'runs'))
    # Sanitize job name for folder
    safe_name = "".join([c if c.isalnum() else "_" for c in job_def['job_name']])
    run_dir = new_run_dir(str(runs_root / safe_name))
    
    # 3. Resume Logic
    start_ep = 1
    opt_state = None
    resume_path = job_def.get('resume_checkpoint')
    
    # Check if we have a resume path OR a local 'latest' checkpoint from a crash
    latest = run_dir / "checkpoint_latest.pt"
    if latest.exists():
        resume_path = str(latest)
        log_cb(f"Found existing progress in run dir, resuming from: {latest}")

    if resume_path and os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location='cpu')
            smart_load_state_dict(model, ckpt.get("model_state_dict", ckpt))
            if "optimizer_state_dict" in ckpt: opt_state = ckpt['optimizer_state_dict']
            if "epoch" in ckpt: start_ep = ckpt['epoch'] + 1
            log_cb(f"Resuming training from epoch {start_ep}")
        except Exception as e:
            log_cb(f"Resume failed ({e}). Starting fresh.")

    # 4. Save Metadata
    extra = {"job_name": job_def['job_name'], "mode": cfg.training_method, "inputs": cfg.inputs_mode}
    write_run_yaml(run_dir, cfg.__dict__, extra)

    # 5. Callbacks
    def save_callback(epoch, net, opt, stats):
        # Save resume point
        state = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch,
            'train_config': cfg.__dict__
        }
        torch.save(state, run_dir / "checkpoint_latest.pt")
        
        # Report progress
        if progress_cb:
            progress_cb(epoch, cfg.epochs, stats)

    # 6. Run Training
    if job_def['method'] == 'pseudo-labeling':
        # PL logic is slightly different (iterative), handled by run_pseudo_label_job logic
        # For simplicity in this unified executor, we delegate back to the PL logic function
        # Note: This means PL won't get per-epoch progress updates on the UI yet, 
        # but standard jobs will.
        run_pseudo_label_job(
            cfg, job_def['job_name'], job_def['labeled_items'], job_def['unlabeled_items'],
            job_def['timm_name'], job_def['in_chans'], str(runs_root),
            job_def['pl_iters'], job_def['pl_thresh'], log_cb
        )
    else:
        model, best = train_model(
            model, job_def['train_items'], job_def['val_items'], 
            cfg, log_cb, start_ep, opt_state, save_callback
        )
        
        # Final Save
        torch.save({
            'model_state_dict': model.state_dict(), 
            'train_config': cfg.__dict__, 
            'best': best
        }, run_dir / "checkpoint.pt")
        
        # Cleanup resume file to save space? Optional. 
        # if latest.exists(): os.remove(latest)
        
        write_metrics(run_dir, best)

# Keep the PL function for the delegate call above
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
            ckpt_src = run_dir / "checkpoint_src.pt"
            torch.save({"model_state_dict": trained_model.state_dict(), "train_config": cfg.__dict__, "best": best}, ckpt_src)
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
                        if xb.shape[-1] != cfg.target_hw: xb = torch.nn.functional.interpolate(xb, size=(cfg.target_hw, cfg.target_hw))
                        xb = xb.to(cfg.device)
                        emb = trained_model(xb)
                        logits = trained_model.classifier(emb)
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        for pi, p in enumerate(probs):
                            it = batch_items[pi]
                            if p >= pl_thresh: new_labeled.append(Item(it.h5_path, it.row_idx, 1, it.cluster))
                            elif p <= (1.0-pl_thresh): new_labeled.append(Item(it.h5_path, it.row_idx, 0, it.cluster))
                            else: remaining.append(it)
                    except: remaining.extend(batch_items)

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