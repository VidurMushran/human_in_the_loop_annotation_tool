# app/utils/model_helpers.py
import os
import torch
from app.ml.models import make_model 

def load_model_from_checkpoint(path: str, device: str = "cpu"):
    """
    Loads a checkpoint, auto-detects input channels from weights, 
    initializes the correct architecture, and returns (model, kind, in_chans).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # 1. Load to CPU first to inspect safely
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 2. Auto-detect input channels from the weight shape of the first layer
    in_chans = 3 # Default
    
    # SimpleCNN first layer name: "feat.0.weight"
    if "feat.0.weight" in state_dict:
        in_chans = state_dict["feat.0.weight"].shape[1]
    # Timm/ResNet often "conv1.weight" or "backbone.conv1.weight"
    elif "backbone.conv1.weight" in state_dict:
        in_chans = state_dict["backbone.conv1.weight"].shape[1]
    elif "conv1.weight" in state_dict:
        in_chans = state_dict["conv1.weight"].shape[1]

    # 3. Read Architecture Config
    t_cfg = ckpt.get("train_config", {})
    kind = t_cfg.get("model_kind", "simple_cnn")
    timm_name = t_cfg.get("timm_name", "resnet18")

    # 4. Initialize Model
    model = make_model(kind=kind, timm_name=timm_name, in_chans=in_chans, n_classes=2)

    # 5. Load Weights
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"[Warn] Strict loading failed, trying non-strict: {e}")
        model.load_state_dict(state_dict, strict=False)

    if device != "cpu":
        model = model.to(device)
        
    model.eval()
    return model, kind, in_chans