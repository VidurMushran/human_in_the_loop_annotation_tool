# app/data/lazy_h5_dataset.py

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class LazyH5Dataset(Dataset):
    def __init__(
        self,
        index_list,
        include_mask=False,
        normalization="none",  # none | per_crop | slide_level
        slide_stats=None,
        transform=None,
    ):
        self.index = index_list
        self.include_mask = include_mask
        self.normalization = normalization
        self.slide_stats = slide_stats
        self.transform = transform
        self._h5_handles = {}

    def _get_file(self, path):
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")
        return self._h5_handles[path]

    def _normalize(self, img, slide_id):
        if self.normalization == "per_crop":
            mean = img.mean(axis=(1,2), keepdims=True)
            std = img.std(axis=(1,2), keepdims=True) + 1e-6
            img = (img - mean) / std

        elif self.normalization == "slide_level" and self.slide_stats:
            stats = self.slide_stats[slide_id]
            img = (img - stats["mean"]) / (stats["std"] + 1e-6)

        return img

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        meta = self.index[i]
        f = self._get_file(meta["h5_path"])

        img = f[meta["img_key"]][meta["idx"]]
        img = np.asarray(img, dtype=np.float32)

        if img.ndim == 2:
            img = img[None, :, :]
        elif img.shape[-1] < 10:  # likely HWC
            img = np.moveaxis(img, -1, 0)

        img = self._normalize(img, meta.get("slide_id"))

        if self.include_mask:
            mask = f[meta["mask_key"]][meta["idx"]]
            mask = mask.astype(np.float32)[None, :, :]
            img = np.concatenate([img, mask], axis=0)

        img = torch.from_numpy(img)

        if self.transform:
            img = self.transform(img)

        label = meta.get("label", -1)
        return img, label
