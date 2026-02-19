# app/training/lightning_module.py

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from app.training.uncertainty import entropy, margin

class CropClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, y)

        ent = entropy(probs).mean()
        mar = margin(probs).mean()

        self.log("val_loss", loss)
        self.log("val_entropy", ent)
        self.log("val_margin", mar)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
