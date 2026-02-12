from __future__ import annotations
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=4, n_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.feat(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

def make_timm_frozen_linear(model_name: str, in_chans: int = 4, n_classes: int = 2):
    try:
        import timm
    except Exception as e:
        raise RuntimeError("timm is not installed; cannot use pretrained backbones.") from e

    m = timm.create_model(model_name, pretrained=True, in_chans=in_chans, num_classes=0)  # no head
    for p in m.parameters():
        p.requires_grad = False

    # timm models output embeddings; create linear head
    dummy = torch.zeros(1, in_chans, 75, 75)
    with torch.no_grad():
        emb = m(dummy)
    emb_dim = emb.shape[-1]
    head = nn.Linear(emb_dim, n_classes)
    return nn.Sequential(m, head)
