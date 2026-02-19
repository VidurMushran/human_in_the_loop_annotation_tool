# app/training/uncertainty.py

import torch
import torch.nn.functional as F

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def margin(probs):
    top2 = torch.topk(probs, 2, dim=1).values
    return top2[:,0] - top2[:,1]
