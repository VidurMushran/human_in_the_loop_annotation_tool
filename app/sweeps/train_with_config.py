# app/sweeps/train_with_config.py

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from app.data.lazy_h5_dataset import LazyH5Dataset
from app.models.model_factory import build_model
from app.training.lightning_module import CropClassifier
import pytorch_lightning as pl

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):

    train_dataset = LazyH5Dataset(
        index_list=...,
        include_mask=cfg.data.include_mask,
        normalization=cfg.data.normalization,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=4,
        shuffle=True,
    )

    model = build_model(
        backbone=cfg.model.backbone,
        in_channels=4 if cfg.data.include_mask else 3,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    )

    lit_model = CropClassifier(model, lr=cfg.training.lr)

    trainer = pl.Trainer(max_epochs=cfg.training.epochs)
    trainer.fit(lit_model, train_loader)

if __name__ == "__main__":
    main()
