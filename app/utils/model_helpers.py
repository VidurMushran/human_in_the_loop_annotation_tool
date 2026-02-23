# app/utils/model_helpers.py
import os
import torch
from app.ml.models import SimpleCNN, make_timm_frozen_linear

def load_model_from_checkpoint(path: str, device: str = "cpu"):
    """
    Loads a checkpoint, auto-detects input channels from weights, 
    initializes the correct architecture, and returns (model, kind, in_chans).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load to CPU first to avoid CUDA OOM if device is busy
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Auto-detect channels from first layer weights
    in_chans = 3
    if "feat.0.weight" in state_dict:
        in_chans = state_dict["feat.0.weight"].shape[1]
    elif "conv1.weight" in state_dict:
        in_chans = state_dict["conv1.weight"].shape[1]
    elif "backbone.conv1.weight" in state_dict:
        in_chans = state_dict["backbone.conv1.weight"].shape[1]

    # Read config
    t_cfg = ckpt.get("train_config", {})
    kind = t_cfg.get("model_kind", "simple_cnn")
    timm_name = t_cfg.get("timm_name", "resnet18")

    # Initialize
    if kind == "simple_cnn":
        model = SimpleCNN(in_ch=in_chans, n_classes=2)
    else:
        model = make_timm_frozen_linear(timm_name, in_chans=in_chans, n_classes=2)

    model.load_state_dict(state_dict)
    if device != "cpu":
        model = model.to(device)
    model.eval()
    
    return model, kind, in_chans