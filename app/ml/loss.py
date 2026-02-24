import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

def get_loss_function(name: str, device="cuda"):
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "weighted_ce":
        # Weight class 1 (junk) higher generally, typically calculated from dataset
        # Defaulting to 1:3 ratio for now, user can tune in code if needed
        weights = torch.tensor([1.0, 3.0]).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    elif name == "focal_loss":
        return FocalLoss()
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()