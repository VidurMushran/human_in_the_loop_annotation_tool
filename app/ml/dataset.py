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
except: TVF = None

@dataclass
class Item:
    h5_path: str
    row_idx: int
    y: int
    cluster: int = -1

class H5StreamDataset(Dataset):
    def __init__(self, items: List[Item], image_key="images", mask_key="masks", feature_key="features", 
                 target_hw=75, aug_flags=(), max_blur_sigma=0.0, seed=0, 
                 inputs_mode="image_only", training_method="supervised"):
        self.items = items
        self.image_key, self.mask_key, self.feature_key = image_key, mask_key, feature_key
        self.target_hw = target_hw
        self.aug_flags = aug_flags
        self.max_blur_sigma = max_blur_sigma
        self.inputs_mode = inputs_mode
        self.training_method = training_method
        self.rng = np.random.default_rng(seed)
        self._handles = {}

    def _open(self, p):
        if p not in self._handles: self._handles[p] = h5py.File(p, "r")
        return self._handles[p]

    def _augment(self, t):
        if not self.aug_flags: return t
        if "hflip" in self.aug_flags and self.rng.random() < 0.5: t = torch.flip(t, [2])
        if "vflip" in self.aug_flags and self.rng.random() < 0.5: t = torch.flip(t, [1])
        if "rotate90" in self.aug_flags and self.rng.random() < 0.5: t = torch.rot90(t, int(self.rng.integers(1,4)), [1,2])
        if "gaussian_blur" in self.aug_flags and self.max_blur_sigma > 0 and self.rng.random() < 0.5 and TVF:
            sigma = self.rng.uniform(0.1, self.max_blur_sigma)
            k = int(2 * round(sigma) + 1); k = max(k, 1)
            if k % 2 == 0: k += 1
            t = TVF.gaussian_blur(t, kernel_size=[k,k], sigma=[sigma,sigma])
        return t

    def __getitem__(self, i):
        it = self.items[i]
        f = self._open(it.h5_path)
        
        # Image
        img = torch.from_numpy(f[self.image_key][it.row_idx]).float().permute(2,0,1) # C,H,W
        
        # Mask
        if "mask" in self.inputs_mode:
            msk = torch.from_numpy(f[self.mask_key][it.row_idx]).float()
            if msk.ndim==2: msk = msk.unsqueeze(0)
            img = torch.cat([img, msk], dim=0)

        # Resize
        if img.shape[-1] != self.target_hw:
            img = F.interpolate(img.unsqueeze(0), size=(self.target_hw,self.target_hw), mode="bilinear").squeeze(0)

        # Features
        feats = torch.tensor([])
        if "features" in self.inputs_mode:
            feats = torch.from_numpy(f[self.feature_key][it.row_idx]).float()

        # Augment & Return
        y = torch.tensor(it.y).long()
        
        if self.training_method == "self-supervised":
            v1 = self._augment(img.clone())
            v2 = self._augment(img.clone())
            return (v1, v2), feats, y
        else:
            return self._augment(img), feats, y

    def __len__(self): return len(self.items)
    def close(self): 
        for h in self._handles.values(): h.close()
        self._handles = {}