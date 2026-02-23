# app/ml/train.py
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
    model_kind: str = "small_cnn"         
    timm_name: str = "resnet18"           
    inputs_mode: str = "image_only"       # 'image_only', 'image_and_mask', 'image_and_features'
    training_method: str = "supervised"   # 'supervised', 'self-supervised', 'pseudo-labeling'
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    device: str = "cuda"
    image_key: str = "images"
    mask_key: str = "masks"               
    target_hw: int = 75
    aug_flags: tuple[str, ...] = ()
    max_blur_sigma: float = 0.0
    seed: int = 0

def nt_xent_loss(z1, z2, temperature=0.5):
    """ Simple Contrastive Loss for Self-Supervised Learning """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    
    sim_matrix = torch.matmul(z, z.T) / temperature
    sim_matrix.fill_diagonal_(-1e9) 
    
    N = z1.size(0)
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z1.device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

@torch.no_grad()
def eval_binary(model, dl, device: str):
    """ Restored Evaluation Function for Supervised Learning """
    model.eval()
    ys = []
    ps = []
    for batch in dl:
        xb = batch[0] if isinstance(batch, (tuple, list)) else batch
        yb = batch[-1] if isinstance(batch, (tuple, list)) else batch[1]
        
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


def train_model(
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
        mask_key=cfg.mask_key,
        target_hw=cfg.target_hw,
        aug_flags=cfg.aug_flags,
        max_blur_sigma=cfg.max_blur_sigma,
        seed=cfg.seed,
        inputs_mode=cfg.inputs_mode,
        training_method=cfg.training_method
    )
    
    val_ds = H5StreamDataset(
        val_items, image_key=cfg.image_key, mask_key=cfg.mask_key, target_hw=cfg.target_hw,
        aug_flags=(), max_blur_sigma=0.0, seed=cfg.seed,
        inputs_mode=cfg.inputs_mode, training_method=cfg.training_method
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    best_metric = -float('inf')
    best_state = None
    t0 = time.time()

    try:
        for ep in range(1, cfg.epochs + 1):
            model.train()
            losses = []

            for batch in train_dl:
                opt.zero_grad(set_to_none=True)
                
                # --- ROUTING BASED ON CONFIG ---
                if cfg.training_method == "self-supervised":
                    (x1, x2), yb = batch
                    x1, x2 = x1.to(device), x2.to(device)
                    z1 = model(x1)
                    z2 = model(x2)
                    loss = nt_xent_loss(z1, z2)
                    
                else: 
                    # Supervised / Pseudo-labeling execution
                    if cfg.inputs_mode == "image_and_features":
                        xb, feats, yb = batch
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb) 
                    else:
                        xb, yb = batch
                        xb, yb = xb.to(device), yb.to(device)
                        logits = model(xb)
                        
                    loss = F.cross_entropy(logits, yb)

                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            # --- VALIDATION EVALUATION ---
            model.eval()
            if cfg.training_method == "self-supervised":
                val_losses = []
                with torch.no_grad():
                    for (x1, x2), yb in val_dl:
                        z1, z2 = model(x1.to(device)), model(x2.to(device))
                        val_losses.append(nt_xent_loss(z1, z2).item())
                avg_val_loss = np.mean(val_losses)
                log_cb(f"[ep {ep}/{cfg.epochs}] Train SSL Loss={np.mean(losses):.4f} Val SSL Loss={avg_val_loss:.4f}")
                
                if -avg_val_loss > best_metric:
                    best_metric = -avg_val_loss
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    
            else:
                acc, auc = eval_binary(model, val_dl, device)
                log_cb(f"[ep {ep}/{cfg.epochs}] loss={np.mean(losses):.4f} val_acc={acc:.3f} val_auc={auc:.3f}")

                if acc > best_metric:
                    best_metric = float(acc)
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        dt = float(time.time() - t0)
        log_cb(f"Done in {dt:.1f}s.")
        if best_state is not None:
            model.load_state_dict(best_state)

        best = {
            "best_metric": float(best_metric),
            "train_time_s": float(dt),
        }
        return model, best

    finally:
        train_ds.close()
        val_ds.close()