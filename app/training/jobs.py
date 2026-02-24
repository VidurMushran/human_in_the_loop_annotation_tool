# app/training/jobs.py
import os
import torch
import yaml
from pathlib import Path
from app.ml.train import train_model, TrainConfig
from app.ml.models import make_model
from app.experiments.registry import write_run_yaml, write_metrics, new_run_dir

def execute_job_from_def(job_def, log_cb):
    cfg = TrainConfig(**job_def['cfg'])
    
    # Init Model
    model = make_model(cfg.model_kind, cfg.timm_name, job_def['in_chans'], 
                       use_features="features" in cfg.inputs_mode,
                       feature_dim=0) # Feature dim would need calculation if using tabular features

    # Paths (Root/Run_Name)
    runs_root = Path(job_def.get('runs_root', 'runs'))
    run_dir = new_run_dir(str(runs_root / job_def['job_name'].replace(" | ", "_").replace(" ", "")))
    
    # Resume Check
    start_ep = 1
    opt_state = None
    latest = run_dir / "checkpoint_latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        start_ep = ckpt['epoch'] + 1
        opt_state = ckpt['optimizer_state_dict']
        log_cb(f"Resuming from epoch {start_ep}")

    # Save Callback
    def save_cb(ep, net, opt, stats):
        state = {'model_state_dict':net.state_dict(), 'optimizer_state_dict':opt.state_dict(), 'epoch':ep}
        torch.save(state, run_dir / "checkpoint_latest.pt")

    # Run
    # (Simplified data loading for this snippet - assuming items passed or re-hydrated)
    # In a real persistent queue, we pass paths and re-create Items, not pickle objects
    train_items = job_def['train_items'] 
    val_items = job_def['val_items']

    model, best = train_model(model, train_items, val_items, cfg, log_cb, start_ep, opt_state, save_cb)
    
    # Final Save
    torch.save({'model_state_dict':model.state_dict(), 'config':cfg.__dict__, 'best':best}, run_dir / "checkpoint.pt")
    write_run_yaml(run_dir, cfg.__dict__, job_def)