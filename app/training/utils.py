# app/training/utils.py
import numpy as np
import h5py
from tqdm import tqdm
import torch

def compute_slide_stats(h5_paths, index_list):
    """
    Compute slide-level mean/std per-channel from a subset or all images.
    index_list: list of metadata dicts (must include 'slide_id', 'h5_path', 'idx')
    returns {slide_id: {'mean': np.array(C,), 'std': np.array(C,) } }
    """
    slides = {}
    grouped = {}
    for item in index_list:
        sid = item.get("slide_id", None)
        if sid is None:
            continue
        grouped.setdefault((item["h5_path"], sid), []).append(item["idx"])

    for (h5p, sid), idxs in grouped.items():
        with h5py.File(h5p, "r") as f:
            imgs = f["images"]
            csum = None
            csq = None
            n = 0
            for i in idxs:
                img = imgs[i].astype(np.float32)
                # HWC -> CHW
                if img.ndim == 2:
                    img = img[..., None]
                if img.shape[-1] <= 8 and img.shape[0] > img.shape[-1]:
                    img = np.moveaxis(img, -1, 0)
                else:
                    img = np.moveaxis(img, -1, 0)  # assume HWC
                if csum is None:
                    csum = img.sum(axis=(1,2))
                    csq = (img ** 2).sum(axis=(1,2))
                else:
                    csum += img.sum(axis=(1,2))
                    csq += (img ** 2).sum(axis=(1,2))
                n += img.shape[1] * img.shape[2]
            # compute mean/std per channel
            mean = csum / n
            var = (csq / n) - (mean ** 2)
            std = np.sqrt(np.maximum(var, 1e-6))
            slides[sid] = {"mean": mean, "std": std}
    return slides

# Temperature scaling (simple)
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = torch.nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temp
