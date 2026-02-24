# app/utils/model_helpers.py
import os
import torch
import logging
from app.ml.models import make_model, SimpleCNN, TimmModel

logger = logging.getLogger(__name__)

def smart_load_state_dict(model, state_dict):
    """
    Loads state_dict into model, ignoring keys that don't match in shape/size.
    Useful for loading checkpoints from slightly different architectures.
    """
    model_state = model.state_dict()
    valid_state = {}
    mismatched = []
    
    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                valid_state[k] = v
            else:
                mismatched.append(k)
        else:
            # Key not in model (e.g. extra layer in checkpoint)
            pass

    if mismatched:
        print(f"[Warn] Dropped {len(mismatched)} mismatched layers: {mismatched[:3]}...")
    
    # strict=False allows loading partial matches
    model.load_state_dict(valid_state, strict=False)

def load_model_from_checkpoint(path: str, device: str = "cpu"):
    """
    Loads a checkpoint, auto-detects input channels/arch, and returns the model.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # 1. Load to CPU first
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # 2. Auto-detect input channels from weights
    in_chans = 3
    if "feat.0.weight" in state_dict: in_chans = state_dict["feat.0.weight"].shape[1]
    elif "backbone.conv1.weight" in state_dict: in_chans = state_dict["backbone.conv1.weight"].shape[1]
    elif "conv1.weight" in state_dict: in_chans = state_dict["conv1.weight"].shape[1]

    # 3. Read Config
    t_cfg = ckpt.get("train_config", {})
    kind = t_cfg.get("model_kind", "simple_cnn")
    timm_name = t_cfg.get("timm_name", "resnet18")

    # 4. Init Model
    model = make_model(kind=kind, timm_name=timm_name, in_chans=in_chans, n_classes=2)

    # 5. Smart Load Weights
    smart_load_state_dict(model, state_dict)

    if device != "cpu":
        model = model.to(device)
    model.eval()
    
    return model, kind, in_chans