# app/models/model_factory.py
import torch
import torch.nn as nn
import timm

class ProjectionHead(nn.Module):
    """ Used for contrastive learning (SimCLR style) """
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def build_model(
    backbone="resnet18", 
    inputs_mode="image_only", 
    training_method="supervised",
    num_classes=2, 
    pretrained=True
):
    # Determine input channels
    in_channels = 3 if inputs_mode == "image_only" else 4 # Assuming mask adds 1 channel

    if backbone == "small_cnn":
        # Modify your SmallCNN to accept dynamic in_channels and return features
        model = SmallCNN(in_ch=in_channels, num_classes=num_classes)
        emb_dim = 128 # Base output of SmallCNN
    else:
        # Create timm model without classifier head first
        model = timm.create_model(backbone, pretrained=pretrained, in_chans=in_channels, num_classes=0)
        
        # Determine embedding dimension by passing a dummy tensor
        dummy = torch.zeros(1, in_channels, 75, 75)
        with torch.no_grad():
            emb = model(dummy)
        emb_dim = emb.shape[-1]

    # Attach the appropriate head based on the training method
    if training_method == "self-supervised":
        head = ProjectionHead(in_dim=emb_dim, out_dim=128)
    else:
        head = nn.Linear(emb_dim, num_classes)

    # Note: If handling "image_and_features", you would return a custom nn.Module 
    # here that overrides `forward` to take `(image, features)`, pass image through 
    # backbone, concat with features, and pass through head. 

    return nn.Sequential(model, head)