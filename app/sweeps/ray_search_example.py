# app/sweeps/ray_search_example.py
from ray import tune
import subprocess
import json
import os

def launch_trial(config):
    # Create a temp config override as JSON or CLI args to hydra entrypoint
    overrides = [
        f"model.backbone={config['model.backbone']}",
        f"model.dropout={config['model.dropout']}",
        f"data.include_mask={str(config['data.include_mask']).lower()}",
        f"training.lr={config['training.lr']}",
        f"training.batch_size={config['training.batch_size']}"
    ]
    cmd = ["python", "app/sweeps/train_with_config.py"] + overrides
    return subprocess.run(cmd).returncode

def run_search():
    search_space = {
        "model.backbone": ["resnet18", "efficientnet_b0", "small_cnn"],
        "model.dropout": [0.0, 0.2, 0.5],
        "data.include_mask": [False, True],
        "training.lr": [1e-4, 3e-4, 1e-3],
        "training.batch_size": [64, 128]
    }
    # naive nested loops for simplicity — use Ray Tune / Slurm integration for scale
    for backbone in search_space["model.backbone"]:
        for dropout in search_space["model.dropout"]:
            for include_mask in search_space["data.include_mask"]:
                for lr in search_space["training.lr"]:
                    for bs in search_space["training.batch_size"]:
                        cfg = {
                            "model": {"backbone": backbone, "dropout": dropout},
                            "data": {"include_mask": include_mask},
                            "training": {"lr": lr, "batch_size": bs}
                        }
                        print("Launching:", cfg)
                        r = launch_trial(cfg)
                        if r != 0:
                            print("Trial failed", cfg)

if __name__ == "__main__":
    run_search()
