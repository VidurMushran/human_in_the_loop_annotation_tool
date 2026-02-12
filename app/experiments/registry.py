from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json, time, yaml

def new_run_dir(root: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    d = Path(root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_run_yaml(run_dir: Path, cfg: dict, extra: dict):
    p = run_dir / "run.yml"
    data = {"config": cfg, **extra}
    p.write_text(yaml.safe_dump(data, sort_keys=False))

def write_metrics(run_dir: Path, metrics: dict):
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

def write_checkpoint(run_dir: Path, src_ckpt_path: str):
    import shutil
    shutil.copy2(src_ckpt_path, run_dir / "checkpoint.pt")
