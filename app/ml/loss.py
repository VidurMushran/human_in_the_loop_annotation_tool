# app/ml/loss.py
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

def nt_xent_loss(z1, z2, temperature=0.5):
    """ 
    Normalized Temperature-scaled Cross Entropy Loss for SSL (SimCLR).
    Expects z1, z2 to be (Batch, Dim).
    """
    device = z1.device
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate: (2N, Dim)
    z = torch.cat([z1, z2], dim=0)
    
    # Cosine similarity matrix: (2N, 2N)
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    # Mask out self-similarity
    sim_matrix.fill_diagonal_(-1e9)
    
    # Targets: z1[i] matches z2[i]. 
    # In the concat tensor, z1 is 0..N-1, z2 is N..2N-1.
    # z1[i] (index i) should match z2[i] (index i+N)
    # z2[i] (index i+N) should match z1[i] (index i)
    N = z1.size(0)
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def get_loss_function(name: str, device="cuda"):
    name = name.lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "weighted_ce":
        # Default 1:3 weighting for class imbalance (Cell:Junk)
        weights = torch.tensor([1.0, 3.0]).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    elif name == "focal_loss":
        return FocalLoss()
    elif name == "bce":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()