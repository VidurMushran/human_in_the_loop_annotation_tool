# app/training/lightning_module.py
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from app.training.uncertainty import entropy_from_logits, margin_from_logits

class CropClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=1)
        ent = entropy_from_logits(logits)
        mar = margin_from_logits(logits)

        # log scalar summaries
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/entropy_mean", ent.mean())
        self.log("val/margin_mean", mar.mean())

        # also log accuracy
        preds = torch.argmax(probs, dim=1)
        acc = (preds == y).float().mean()
        self.log("val/acc", acc, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # returns logits, probs, entropy, margin
        x, meta = batch  # for predict we expect (img_tensor, meta_dict) or adapt call
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        ent = entropy_from_logits(logits)
        mar = margin_from_logits(logits)
        return {"logits": logits.detach().cpu(), "probs": probs.detach().cpu(), "entropy": ent.detach().cpu(), "margin": mar.detach().cpu()}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt
