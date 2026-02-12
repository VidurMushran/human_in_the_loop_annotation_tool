from __future__ import annotations
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
import h5py

from app.data.h5io import write_features_column_inplace


@torch.no_grad()
def score_h5_file(
    model,
    h5_path: str,
    score_col: str,
    image_key: str,
    features_key: str,
    device: str,
    batch_size: int,
    target_hw: int,
    log_cb: Callable[[str], None],
):
    device = device if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    probs = []
    with h5py.File(h5_path, "r") as f:
        X = f[image_key]
        n = int(X.shape[0])

        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            arr = X[s:e].astype(np.float32)                 # NHWC
            xb = torch.from_numpy(arr).permute(0, 3, 1, 2)  # NCHW

            if xb.shape[-2] != target_hw or xb.shape[-1] != target_hw:
                xb = F.interpolate(xb, size=(target_hw, target_hw), mode="bilinear", align_corners=False)

            xb = xb.to(device, non_blocking=True)
            p1 = torch.softmax(model(xb), dim=1)[:, 1].detach().cpu().numpy()
            probs.append(p1)

    probs = np.concatenate(probs).astype(np.float32)
    write_features_column_inplace(h5_path, score_col, probs, features_key=features_key)
    log_cb(f"Scored {_basename(h5_path) if ' _basename' in globals() else h5_path} -> wrote '{score_col}' (n={len(probs)})")
