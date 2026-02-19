# app/models/model_factory.py
import torch.nn as nn
import timm

class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        f = self.encoder(x)
        return self.head(f.view(f.size(0), -1))

def build_model(backbone="resnet18", in_channels=3, num_classes=2, pretrained=True, dropout=0.2):
    if backbone == "small_cnn":
        return SmallCNN(in_ch=in_channels, num_classes=num_classes, dropout=dropout)
    else:
        # timm handles in_chans and num_classes
        model = timm.create_model(backbone, pretrained=pretrained, in_chans=in_channels, num_classes=num_classes)
        return model
