# app/data/lazy_h5_dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class LazyH5Dataset(Dataset):
    """
    index_list: list of dicts with keys:
      - h5_path, img_key ('images'), idx (image index)
      - slide_id (optional), label (optional), mask_key (optional)
    include_mask: if True, will try to read mask dataset named 'masks' in same HDF5
    normalization: 'none' | 'per_crop' | 'slide_level'
    slide_stats: dict mapping slide_id -> {'mean': np.array(C), 'std': np.array(C)}
    transform: callable that accepts torch.Tensor (C,H,W) and returns torch.Tensor
    """
    def __init__(self, index_list, include_mask=False, normalization="none", slide_stats=None, transform=None):
        self.index = index_list
        self.include_mask = include_mask
        self.normalization = normalization
        self.slide_stats = slide_stats or {}
        self.transform = transform
        self._h5_handles = {}  # per-worker handles

    def _open(self, path):
        # Open file per-process / per-worker
        if path not in self._h5_handles:
            # Use swmr=False write-safe reading; open read-only
            self._h5_handles[path] = h5py.File(path, "r")
        return self._h5_handles[path]

    def _to_chw(self, arr):
        # arr: H x W x C or H x W
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.shape[-1] <= 8 and arr.shape[0] > arr.shape[-1]:
            # likely HWC, convert
            arr = np.moveaxis(arr, -1, 0)
        else:
            # already CHW or ambiguous; try to ensure CHW
            if arr.shape[0] <= 8 and arr.shape[-1] > 8:
                arr = np.moveaxis(arr, -1, 0)
        return arr

    def _normalize(self, img, slide_id=None):
        # img: np.array (C,H,W)
        if self.normalization == "per_crop":
            mean = img.mean(axis=(1,2), keepdims=True)
            std = img.std(axis=(1,2), keepdims=True) + 1e-6
            img = (img - mean) / std
        elif self.normalization == "slide_level" and slide_id in self.slide_stats:
            stats = self.slide_stats[slide_id]
            mean = stats["mean"][:, None, None]
            std = stats["std"][:, None, None] + 1e-6
            img = (img - mean) / std
        return img

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        meta = self.index[i]
        h5p = meta["h5_path"]
        f = self._open(h5p)

        img_ds = f[meta.get("img_key", "images")]
        img = img_ds[meta["idx"]]
        img = np.asarray(img, dtype=np.float32)  # uint16 -> float32

        img = self._to_chw(img)  # ensure (C,H,W)

        img = self._normalize(img, meta.get("slide_id"))

        if self.include_mask:
            mask_key = meta.get("mask_key", "masks")
            if mask_key in f:
                mask_ds = f[mask_key]
                mask = np.asarray(mask_ds[meta["idx"]], dtype=np.float32)
                if mask.ndim == 2:
                    mask = mask[np.newaxis, ...]
                else:
                    mask = self._to_chw(mask)
                # concatenate mask as extra channel
                img = np.concatenate([img, mask], axis=0)
            else:
                # no mask available: append zeros channel
                h, w = img.shape[1], img.shape[2]
                zero_m = np.zeros((1, h, w), dtype=np.float32)
                img = np.concatenate([img, zero_m], axis=0)

        # convert to torch tensor
        img_t = torch.from_numpy(img)

        if self.transform is not None:
            img_t = self.transform(img_t)

        label = meta.get("label", -1)
        return img_t, int(label) if label is not None and label != -1 else -1
