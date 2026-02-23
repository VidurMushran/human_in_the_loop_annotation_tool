# app/ml/dataset.py
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
        mask_key: str = "masks",       # Added for image_and_mask
        feature_key: str = "features", # Added for image_and_features
        target_hw: int = 75,
        aug_flags: tuple[str, ...] = (),
        max_blur_sigma: float = 0.0,
        seed: int = 0,
        inputs_mode: str = "image_only",       # 'image_only', 'image_and_mask', 'image_and_features'
        training_method: str = "supervised"    # 'supervised', 'self-supervised'
    ):
        self.items = items
        self.image_key = image_key
        self.mask_key = mask_key
        self.feature_key = feature_key
        self.target_hw = int(target_hw)
        self.aug_flags = tuple(aug_flags or ())
        self.max_blur_sigma = float(max_blur_sigma or 0.0)
        self.inputs_mode = inputs_mode
        self.training_method = training_method
        self._handles = {}
        self._rng = np.random.default_rng(int(seed))

    def __len__(self):
        return len(self.items)

    def _open(self, path):
        if path in self._handles:
            return self._handles[path]
        f = h5py.File(path, "r")
        self._handles[path] = f
        return f

    def _augment(self, x_chw: torch.Tensor) -> torch.Tensor:
        if not self.aug_flags:
            return x_chw

        # Clone to prevent modifying the base tensor when generating multiple views
        x_aug = x_chw.clone() 
        
        if "hflip" in self.aug_flags and self._rng.random() < 0.5:
            x_aug = torch.flip(x_aug, dims=[2])
        if "vflip" in self.aug_flags and self._rng.random() < 0.5:
            x_aug = torch.flip(x_aug, dims=[1])
        if "rotate90" in self.aug_flags and self._rng.random() < 0.5:
            k = int(self._rng.integers(0, 4))
            x_aug = torch.rot90(x_aug, k=k, dims=(1, 2))
        if ("gaussian_blur" in self.aug_flags and self.max_blur_sigma > 0 
            and self._rng.random() < 0.5 and TVF is not None):
            sigma = float(self._rng.random() * self.max_blur_sigma)
            k = max(3, int(2 * round(2 * sigma) + 1))
            x_aug = TVF.gaussian_blur(x_aug, kernel_size=[k, k], sigma=[sigma, sigma])

        return x_aug

    def __getitem__(self, i):
        it = self.items[i]
        f = self._open(it.h5_path)
        
        # 1. Load Base Image
        arr = f[self.image_key][int(it.row_idx)].astype(np.float32)  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        
        # 2. Add extra modalities based on inputs_mode
        if self.inputs_mode == "image_and_mask":
            # Assuming mask is HW or HWC, we add it as an extra channel
            mask_arr = f[self.mask_key][int(it.row_idx)].astype(np.float32)
            if len(mask_arr.shape) == 2:
                mask_arr = np.expand_dims(mask_arr, axis=-1)
            mask_t = torch.from_numpy(mask_arr).permute(2, 0, 1).unsqueeze(0)
            t = torch.cat([t, mask_t], dim=1) # Concatenate along channels
            
        elif self.inputs_mode == "image_and_features":
            # If combining with 1D features, you'll likely want to return them separately 
            # so the model can route them to a fully connected layer later.
            feat_arr = f[self.feature_key][int(it.row_idx)].astype(np.float32)
            features = torch.from_numpy(feat_arr)

        # 3. Interpolate Spatial Data
        if (t.shape[-2], t.shape[-1]) != (self.target_hw, self.target_hw):
            t = F.interpolate(t, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        x_base = t.squeeze(0)  # C,H,W

        y = torch.tensor(int(it.y), dtype=torch.long)

        # 4. Routing based on training method
        if self.training_method == "self-supervised":
            # Return two differently augmented views of the same data for contrastive loss
            x_view1 = self._augment(x_base)
            x_view2 = self._augment(x_base)
            if self.inputs_mode == "image_and_features":
                return (x_view1, x_view2), features, y
            return (x_view1, x_view2), y

        else:
            # Supervised: standard single view
            x = self._augment(x_base)
            if self.inputs_mode == "image_and_features":
                return x, features, y
            return x, y

    def close(self):
        for _, f in self._handles.items():
            try:
                f.close()
            except Exception:
                pass
        self._handles.clear()