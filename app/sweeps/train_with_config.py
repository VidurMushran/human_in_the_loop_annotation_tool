# app/sweeps/train_with_config.py
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from app.data.index_builder import build_index_from_h5
from app.data.lazy_h5_dataset import LazyH5Dataset
from app.models.model_factory import build_model
from app.training.lightning_module import CropClassifier
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import os

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n", OmegaConf.to_yaml(cfg))

    # build index from HDF5 path(s)
    # Here we assume index_path points to a single HDF5 for simplicity; you can extend to JSON index as needed.
    index = build_index_from_h5(cfg.data.index_path, require_label=True)
    print(f"Index size: {len(index)}")

    dataset = LazyH5Dataset(
        index,
        include_mask=cfg.data.include_mask,
        normalization=cfg.data.normalization,
        transform=None
    )

    # simple train/val split by slide or random
    # For now do random split
    n = len(dataset)
    idxs = list(range(n))
    split = int(0.8 * n)
    train_idx, val_idx = idxs[:split], idxs[split:]

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True)

    # instantiate model
    in_ch =  (4 if cfg.data.include_mask else None)
    # infer in_ch from a sample:
    sample_x, _ = dataset[0]
    in_ch = sample_x.shape[0]
    model = build_model(backbone=cfg.model.backbone, in_channels=in_ch, num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained, dropout=cfg.model.dropout)

    lit = CropClassifier(model, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name="runs")
    trainer = pl.Trainer(max_epochs=cfg.training.epochs, logger=logger, devices=1 if torch.cuda.is_available() else None, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(lit, train_loader, val_loader)

if __name__ == "__main__":
    main()
