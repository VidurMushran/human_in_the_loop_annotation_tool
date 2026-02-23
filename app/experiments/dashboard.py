# app/experiments/dashboard.py
import os
import yaml
import json

def parse_runs_directory(runs_dir: str):
    """
    Scans a directory for run folders, parses run.yml and metrics.json.
    Returns a list of dictionaries with run metadata.
    """
    all_runs = []
    if not runs_dir or not os.path.exists(runs_dir):
        return all_runs

    for folder_name in os.listdir(runs_dir):
        folder_path = os.path.join(runs_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        run_yml = os.path.join(folder_path, "run.yml")
        metrics_json = os.path.join(folder_path, "metrics.json")
        
        if not os.path.exists(run_yml):
            continue

        try:
            with open(run_yml, 'r') as f:
                run_data = yaml.safe_load(f) or {}
            
            cfg = run_data.get("cfg", {})
            extra = run_data.get("extra", {})
            
            # Robust extraction of metadata
            method = extra.get("mode") or cfg.get("training_method")
            if not method and "supervised" in folder_name: method = "supervised"
            if not method: method = "unknown"

            input_mode = extra.get("inputs") or cfg.get("inputs_mode")
            if not input_mode and "image" in folder_name: input_mode = "image_only"
            if not input_mode: input_mode = "unknown"

            arch = cfg.get("timm_name") if cfg.get("model_kind") == "timm_frozen" else "simple_cnn"
            
            # Metrics
            metric_val = 0.0
            time_s = 0.0
            if os.path.exists(metrics_json):
                with open(metrics_json, 'r') as f:
                    m_data = json.load(f)
                    # Support various metric keys
                    metric_val = m_data.get("best_metric", m_data.get("acc", m_data.get("best", 0.0)))
                    if isinstance(metric_val, str):
                        try:
                            metric_val = float(metric_val.split('(')[-1].split(')')[0])
                        except:
                            metric_val = 0.0
                    time_s = m_data.get("train_time_s", 0.0)

            all_runs.append({
                "folder_name": folder_name,
                "folder_path": folder_path,
                "method": str(method),
                "inputs": str(input_mode),
                "arch": str(arch),
                "metric": float(metric_val),
                "time": float(time_s)
            })
        except Exception as e:
            print(f"Error parsing {folder_name}: {e}")
            
    return all_runs