# app/training/headless_main.py
import argparse
import yaml
import sys
import os
import torch
import pickle
from app.training.jobs import run_standard_job, run_pseudo_label_job
from app.ml.train import TrainConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to sweep_config.pkl")
    args = parser.parse_args()

    print(f"--- Headless Training Worker ---")
    print(f"Loading config: {args.config}")

    try:
        with open(args.config, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"FATAL: Could not load config pickle: {e}")
        return

    jobs = data.get("jobs", [])
    runs_root = data.get("runs_root", "runs")
    
    print(f"Found {len(jobs)} jobs to run.")
    print(f"Output directory: {runs_root}")

    for i, job in enumerate(jobs):
        print(f"\n[Job {i+1}/{len(jobs)}] {job['job_name']}")
        
        cfg = job['cfg']
        # Rehydrate config object
        t_cfg = TrainConfig(**cfg)
        
        log_cb = lambda s: print(f"[{job['job_name']}] {s}", flush=True)
        
        try:
            if job['method'] == "pseudo-labeling":
                run_pseudo_label_job(
                    cfg=t_cfg,
                    job_name=job['job_name'],
                    init_labeled=job['labeled_items'],
                    init_unlabeled=job['unlabeled_items'],
                    timm_name=job['timm_name'],
                    in_chans=job['in_chans'],
                    runs_root=runs_root,
                    pl_iters=job['pl_iters'],
                    pl_thresh=job['pl_thresh'],
                    log_cb=log_cb
                )
            else:
                run_standard_job(
                    cfg=t_cfg,
                    job_name=job['job_name'],
                    train_items=job['train_items'],
                    val_items=job['val_items'],
                    timm_name=job['timm_name'],
                    in_chans=job['in_chans'],
                    runs_root=runs_root,
                    log_cb=log_cb,
                    resume_checkpoint=job.get("resume_checkpoint")
                )
            print(f"[Job {i+1}] Finished Successfully.")
        except Exception as e:
            print(f"[Job {i+1}] FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()