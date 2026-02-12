from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import h5py

try:
    import torchvision.transforms.functional as TVF
except Exception:
    TVF = None


@dataclass
class Item:
    h5_path: str
    row_idx: int
    y: int
    cluster: int = -1


class H5StreamDataset(Dataset):
    def __init__(
        self,
        items: List[Item],
        image_key: str = "images",
        target_hw: int = 75,
        aug_flags: tuple[str, ...] = (),
        max_blur_sigma: float = 0.0,
        seed: int = 0,
    ):
        self.items = items
        self.image_key = image_key
        self.target_hw = int(target_hw)
        self.aug_flags = tuple(aug_flags or ())
        self.max_blur_sigma = float(max_blur_sigma or 0.0)
        self._handles = {}
        self._rng = np.random.default_rng(int(seed))

    def __len__(self):
        return len(self.items)

    def _open(self, path):
        if path in self._handles:
            return self._handles[path]
        f = h5py.File(path, "r")
        d = f[self.image_key]
        self._handles[path] = (f, d)
        return self._handles[path]

    def _augment(self, x_chw: torch.Tensor) -> torch.Tensor:
        if not self.aug_flags:
            return x_chw

        if "hflip" in self.aug_flags and self._rng.random() < 0.5:
            x_chw = torch.flip(x_chw, dims=[2])
        if "vflip" in self.aug_flags and self._rng.random() < 0.5:
            x_chw = torch.flip(x_chw, dims=[1])

        if "rotate90" in self.aug_flags and self._rng.random() < 0.5:
            k = int(self._rng.integers(0, 4))
            x_chw = torch.rot90(x_chw, k=k, dims=(1, 2))

        if (
            "gaussian_blur" in self.aug_flags
            and self.max_blur_sigma > 0
            and self._rng.random() < 0.5
            and TVF is not None
        ):
            sigma = float(self._rng.random() * self.max_blur_sigma)
            k = max(3, int(2 * round(2 * sigma) + 1))  # odd
            x_chw = TVF.gaussian_blur(x_chw, kernel_size=[k, k], sigma=[sigma, sigma])

        return x_chw

    def __getitem__(self, i):
        it = self.items[i]
        f, d = self._open(it.h5_path)
        arr = d[int(it.row_idx)].astype(np.float32)  # HWC

        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        if (t.shape[-2], t.shape[-1]) != (self.target_hw, self.target_hw):
            t = F.interpolate(t, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        x = t.squeeze(0)  # C,H,W
        x = self._augment(x)

        y = torch.tensor(int(it.y), dtype=torch.long)
        return x, y

    def close(self):
        for _, (f, _) in self._handles.items():
            try:
                f.close()
            except Exception:
                pass
        self._handles.clear()
