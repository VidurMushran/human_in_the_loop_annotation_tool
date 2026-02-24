# app/ml/models.py
import torch
import torch.nn as nn
import timm

class SimpleCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, embed_dim=128):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        # Projection head (128 -> 128)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim) 
        )
        # Classifier head
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, features=None):
        # x: images, features: optional 1D vector
        x = self.feat(x)
        emb = self.projector(x)
        
        if features is not None:
            # Placeholder for future feature fusion if desired
            pass
            
        return emb  # Return embedding. Caller applies classifier if needed.

class TimmModel(nn.Module):
    def __init__(self, name, in_chans=3, n_classes=2, embed_dim=128):
        super().__init__()
        # Create backbone without classifier
        self.backbone = timm.create_model(name, pretrained=True, in_chans=in_chans, num_classes=0)
        
        # Determine output dimension dynamically
        with torch.no_grad():
            dummy = torch.randn(1, in_chans, 64, 64)
            out_dim = self.backbone(dummy).shape[1]

        # Projector for SSL/Embedding
        self.projector = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
        # Linear Classifier
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, features=None):
        x = self.backbone(x)
        emb = self.projector(x)
        return emb

def make_model(kind: str, timm_name: str, in_chans: int, n_classes: int = 2):
    """
    Factory function to create models.
    kind: 'simple_cnn' or 'timm_frozen' (or just 'timm')
    """
    if kind == "simple_cnn":
        return SimpleCNN(in_ch=in_chans, n_classes=n_classes)
    else:
        # 'timm_frozen' or 'timm' -> TimmModel
        # Note: We aren't strictly freezing layers here, allowing full finetuning.
        return TimmModel(name=timm_name, in_chans=in_chans, n_classes=n_classes)