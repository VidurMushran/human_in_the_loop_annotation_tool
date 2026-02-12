from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from app.ml.dataset import H5StreamDataset, Item


@dataclass
class TrainConfig:
    model_kind: str  # "simple_cnn" | "timm_frozen"
    timm_name: str = "resnet18"
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "cuda"
    image_key: str = "images"
    target_hw: int = 75

    # augmentations (used by dataset)
    aug_flags: tuple[str, ...] = ()
    max_blur_sigma: float = 0.0

    seed: int = 0


@torch.no_grad()
def eval_binary(model, dl, device: str):
    model.eval()
    ys = []
    ps = []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        prob1 = torch.softmax(model(xb), dim=1)[:, 1]
        ys.append(yb.detach().cpu().numpy())
        ps.append(prob1.detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return float(acc), float(auc)


def train_supervised(
    model,
    train_items: List[Item],
    val_items: List[Item],
    cfg: TrainConfig,
    log_cb: Callable[[str], None],
) -> Tuple[torch.nn.Module, Dict]:
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    model = model.to(device)

    train_ds = H5StreamDataset(
        train_items,
        image_key=cfg.image_key,
        target_hw=cfg.target_hw,
        aug_flags=cfg.aug_flags,
        max_blur_sigma=cfg.max_blur_sigma,
        seed=cfg.seed,
    )
    val_ds = H5StreamDataset(
        val_items,
        image_key=cfg.image_key,
        target_hw=cfg.target_hw,
        aug_flags=(),
        max_blur_sigma=0.0,
        seed=cfg.seed,
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    best_acc = -1.0
    best_auc = float("nan")
    best_epoch = -1
    best_state = None

    t0 = time.time()

    try:
        for ep in range(1, cfg.epochs + 1):
            model.train()
            losses = []

            for xb, yb in train_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            acc, auc = eval_binary(model, val_dl, device)
            log_cb(f"[ep {ep}/{cfg.epochs}] loss={np.mean(losses):.4f} val_acc={acc:.3f} val_auc={auc:.3f}")

            if acc > best_acc:
                best_acc = float(acc)
                best_auc = float(auc)
                best_epoch = int(ep)
                # keep tensor state internal (not JSON)
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        dt = float(time.time() - t0)
        log_cb(f"Done in {dt:.1f}s. Best val_acc={best_acc:.3f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        # JSON-safe best dict
        best = {
            "acc": float(best_acc),
            "auc": float(best_auc) if np.isfinite(best_auc) else float("nan"),
            "best_epoch": int(best_epoch),
            "train_time_s": float(dt),
        }
        return model, best

    finally:
        train_ds.close()
        val_ds.close()
