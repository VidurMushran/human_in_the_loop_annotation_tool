from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass
from app.ml.dataset import H5StreamDataset
from app.ml.loss import get_loss_function, nt_xent_loss
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

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
    aug_flags: tuple = ()
    max_blur_sigma: float = 0.0
    seed: int = 0

def train_model(model, train_items, val_items, cfg: TrainConfig, log_cb, start_epoch=1, optimizer_state=None, save_cb=None):
    device = cfg.device
    model = model.to(device)
    
    # Loss
    criterion = get_loss_function(cfg.loss_function, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if optimizer_state: optimizer.load_state_dict(optimizer_state)

    # Data
    ds_kws = dict(image_key=cfg.image_key, mask_key=cfg.mask_key, feature_key=cfg.feature_key, 
                  target_hw=cfg.target_hw, inputs_mode=cfg.inputs_mode, training_method=cfg.training_method)
    
    train_ds = H5StreamDataset(train_items, aug_flags=cfg.aug_flags, max_blur_sigma=cfg.max_blur_sigma, **ds_kws)
    val_ds = H5StreamDataset(val_items, aug_flags=(), max_blur_sigma=0, **ds_kws)
    
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    best_metric = -float('inf')
    best_state = None

    for ep in range(start_epoch, cfg.epochs+1):
        model.train()
        epoch_losses = []
        
        for batch in train_dl:
            optimizer.zero_grad()
            
            if cfg.training_method == "self-supervised":
                (v1, v2), feats, _ = batch
                z1 = model(v1.to(device), feats.to(device))
                z2 = model(v2.to(device), feats.to(device))
                loss = nt_xent_loss(z1, z2)
            else:
                img, feats, y = batch
                emb = model(img.to(device), feats.to(device)) # 128D
                logits = model.classifier(emb) # 2D
                loss = criterion(logits, y.to(device))
            
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Eval
        model.eval()
        val_metric = 0
        with torch.no_grad():
            if cfg.training_method == "self-supervised":
                vl = []
                for (v1,v2), f, _ in val_dl:
                    z1 = model(v1.to(device), f.to(device))
                    z2 = model(v2.to(device), f.to(device))
                    vl.append(nt_xent_loss(z1, z2).item())
                val_metric = -np.mean(vl)
                log_cb(f"[Ep {ep}] Train Loss: {np.mean(epoch_losses):.3f} | Val SSL Loss: {np.mean(vl):.3f}")
            else:
                ys, ps = [], []
                for img, f, y in val_dl:
                    emb = model(img.to(device), f.to(device))
                    logits = model.classifier(emb)
                    ys.append(y.cpu().numpy())
                    ps.append(torch.softmax(logits, 1)[:,1].cpu().numpy())
                
                y = np.concatenate(ys); p = np.concatenate(ps)
                acc = accuracy_score(y, p>=0.5)
                try: auc = roc_auc_score(y, p)
                except: auc = 0.0
                val_metric = acc
                log_cb(f"[Ep {ep}] Loss: {np.mean(epoch_losses):.3f} | Acc: {acc:.3f} | AUC: {auc:.3f}")

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if save_cb: save_cb(ep, model, optimizer, {"metric": val_metric})

    if best_state: model.load_state_dict(best_state)
    return model, {"best_metric": float(best_metric)}