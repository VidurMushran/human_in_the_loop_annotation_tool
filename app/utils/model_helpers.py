# app/utils/model_helpers.py
import os
import torch
import logging
from app.ml.models import make_model 

logger = logging.getLogger(__name__)

def smart_load_state_dict(model, state_dict):
    """Safe loading that reports specific mismatches without crashing."""
    model_state = model.state_dict()
    valid_state = {}
    mismatched = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                valid_state[k] = v
            else:
                mismatched.append(f"{k} {v.shape} vs {model_state[k].shape}")
        
    if mismatched:
        print(f"[Warn] Dropped {len(mismatched)} mismatched layers: {mismatched[:3]}...")
    
    model.load_state_dict(valid_state, strict=False)

def load_model_from_checkpoint(path: str, device: str = "cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # 1. Load to CPU
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 2. Auto-detect input channels
    in_chans = 3
    if "feat.0.weight" in state_dict: in_chans = state_dict["feat.0.weight"].shape[1]
    elif "backbone.conv1.weight" in state_dict: in_chans = state_dict["backbone.conv1.weight"].shape[1]
    elif "conv1.weight" in state_dict: in_chans = state_dict["conv1.weight"].shape[1]

    t_cfg = ckpt.get("train_config", {})
    kind = t_cfg.get("model_kind", "simple_cnn")
    timm_name = t_cfg.get("timm_name", "resnet18")

    # 3. ARCHITECTURE DETECTION (Fix for "Exact Model" loading)
    # If it's a SimpleCNN but the checkpoint has NO projector, it's a Legacy model.
    if kind == "simple_cnn" and "projector.0.weight" not in state_dict:
        print(f"Detected LEGACY SimpleCNN checkpoint (no projector). Loading legacy architecture.")
        kind = "legacy_simple_cnn"

    # 4. Init correct model
    model = make_model(kind=kind, timm_name=timm_name, in_chans=in_chans, n_classes=2)

    # 5. Load
    smart_load_state_dict(model, state_dict)

    if device != "cpu":
        model = model.to(device)
    model.eval()
    
    return model, kind, in_chans