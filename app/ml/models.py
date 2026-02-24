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
        # Projection for SSL or Dense for Supervised
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim) 
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, features=None):
        x = self.feat(x)
        x = self.projector(x)
        if features is not None:
            # Simple fusion: concat features to embedding (naive) or just return embedding
            # For strict "work from 128", we assume features were handled before or ignored here
            # Ideally fusion happens before projector, but for SimpleCNN we keep it simple.
            pass
        return x  # Returns 128D embedding. Loss function wrapper handles classification.

class TimmModel(nn.Module):
    def __init__(self, name, in_chans=3, n_classes=2, embed_dim=128, use_features=False, feature_dim=0):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, in_chans=in_chans, num_classes=0)
        
        # Get backbone output dim
        with torch.no_grad():
            dummy = torch.randn(1, in_chans, 64, 64)
            out_dim = self.backbone(dummy).shape[1]

        self.use_features = use_features
        if use_features:
            self.feat_mlp = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU())
            fusion_dim = out_dim + 64
        else:
            fusion_dim = out_dim

        # Projector (SimCLR style: 128D)
        self.projector = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        
        # Classifier (Linear Probe on top of 128D)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x, features=None):
        emb = self.backbone(x)
        
        if self.use_features and features is not None:
            f_emb = self.feat_mlp(features)
            emb = torch.cat([emb, f_emb], dim=1)
            
        proj = self.projector(emb) # 128D
        return proj # Return embedding. Training loop calls classifier(proj) if supervised.

def make_model(kind, timm_name, in_chans, n_classes=2, use_features=False, feature_dim=0):
    if kind == "simple_cnn":
        return SimpleCNN(in_ch=in_chans, n_classes=n_classes)
    else:
        return TimmModel(timm_name, in_chans, n_classes, use_features=use_features, feature_dim=feature_dim)