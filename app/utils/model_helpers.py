# app/utils/model_helpers.py
import os
import torch
from app.ml.models import make_model 

def load_model_from_checkpoint(path: str, device: str = "cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Auto-detect input channels
    in_chans = 3
    if "feat.0.weight" in state_dict: in_chans = state_dict["feat.0.weight"].shape[1]
    elif "backbone.conv1.weight" in state_dict: in_chans = state_dict["backbone.conv1.weight"].shape[1]
    elif "conv1.weight" in state_dict: in_chans = state_dict["conv1.weight"].shape[1]

    t_cfg = ckpt.get("train_config", {})
    kind = t_cfg.get("model_kind", "simple_cnn")
    timm_name = t_cfg.get("timm_name", "resnet18")

    # Use Factory
    model = make_model(kind=kind, timm_name=timm_name, in_chans=in_chans, n_classes=2)

    try: model.load_state_dict(state_dict)
    except: model.load_state_dict(state_dict, strict=False)

    if device != "cpu": model = model.to(device)
    model.eval()
    return model, kind, in_chans