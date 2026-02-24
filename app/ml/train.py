# app/ml/train.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from app.ml.dataset import H5StreamDataset, Item
from app.ml.loss import get_loss_function, nt_xent_loss # Now works

@dataclass
class TrainConfig:
    model_kind: str = "simple_cnn"         
    timm_name: str = "resnet18"           
    inputs_mode: str = "image_only"
    training_method: str = "supervised"
    loss_function: str = "cross_entropy"
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "cuda"
    image_key: str = "images"
    mask_key: str = "masks"
    feature_key: str = "features"
    target_hw: int = 75
    aug_flags: tuple[str, ...] = ()
    max_blur_sigma: float = 0.0
    seed: int = 0

@torch.no_grad()
def eval_binary(model, dl, device: str):
    model.eval()
    ys, ps = [], []
    for batch in dl:
        # Unpack based on dataset return signature: img, feats, y
        img, feats, y = batch
        img = img.to(device, non_blocking=True)
        feats = feats.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        emb = model(img, feats)
        logits = model.classifier(emb)
        prob1 = torch.softmax(logits, dim=1)[:, 1]
        
        ys.append(y.cpu().numpy())
        ps.append(prob1.cpu().numpy())
        
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = accuracy_score(y, (p >= 0.5).astype(int))
    try: auc = roc_auc_score(y, p)
    except: auc = float("nan")
    return float(acc), float(auc)

def train_model(
    model, train_items, val_items, cfg: TrainConfig, log_cb, 
    start_epoch=1, optimizer_state=None, save_cb=None
):
    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    model = model.to(device)
    criterion = get_loss_function(cfg.loss_function, device)
    
    ds_kws = dict(image_key=cfg.image_key, mask_key=cfg.mask_key, feature_key=cfg.feature_key,
                  target_hw=cfg.target_hw, inputs_mode=cfg.inputs_mode, training_method=cfg.training_method)

    train_ds = H5StreamDataset(train_items, aug_flags=cfg.aug_flags, max_blur_sigma=cfg.max_blur_sigma, seed=cfg.seed, **ds_kws)
    val_ds = H5StreamDataset(val_items, aug_flags=(), max_blur_sigma=0, seed=cfg.seed, **ds_kws)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if optimizer_state: opt.load_state_dict(optimizer_state)

    best_metric = -float('inf')
    best_state = None
    t0 = time.time()

    try:
        for ep in range(start_epoch, cfg.epochs + 1):
            model.train()
            losses = []
            for batch in train_dl:
                opt.zero_grad(set_to_none=True)
                
                if cfg.training_method == "self-supervised":
                    (v1, v2), feats, _ = batch
                    z1 = model(v1.to(device), feats.to(device))
                    z2 = model(v2.to(device), feats.to(device))
                    loss = nt_xent_loss(z1, z2)
                else:
                    img, feats, y = batch
                    emb = model(img.to(device), feats.to(device))
                    logits = model.classifier(emb)
                    loss = criterion(logits, y.to(device))

                loss.backward()
                opt.step()
                losses.append(loss.item())

            # Validation
            model.eval()
            stats = {}
            if cfg.training_method == "self-supervised":
                vl = []
                with torch.no_grad():
                    for (v1, v2), f, _ in val_dl:
                        z1 = model(v1.to(device), f.to(device))
                        z2 = model(v2.to(device), f.to(device))
                        vl.append(nt_xent_loss(z1, z2).item())
                avg_val_loss = np.mean(vl)
                log_cb(f"[Ep {ep}] Loss: {np.mean(losses):.3f} | Val SSL: {avg_val_loss:.3f}")
                metric = -avg_val_loss
                stats["val_loss"] = avg_val_loss
            else:
                acc, auc = eval_binary(model, val_dl, device)
                log_cb(f"[Ep {ep}] Loss: {np.mean(losses):.3f} | Acc: {acc:.3f} | AUC: {auc:.3f}")
                metric = float(acc)
                stats.update({"acc": acc, "auc": auc})

            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            if save_cb: save_cb(ep, model, opt, stats)

        dt = time.time() - t0
        log_cb(f"Done in {dt:.1f}s")
        if best_state: model.load_state_dict(best_state)
        return model, {"best_metric": float(best_metric)}

    finally:
        train_ds.close()
        val_ds.close()