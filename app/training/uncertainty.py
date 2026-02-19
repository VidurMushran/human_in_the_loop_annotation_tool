# app/training/uncertainty.py
import torch
import torch.nn.functional as F

def softmax_probs(logits):
    return F.softmax(logits, dim=1)

def entropy_from_logits(logits):
    p = softmax_probs(logits)
    ent = -torch.sum(p * torch.log(p + 1e-12), dim=1)
    return ent

def margin_from_logits(logits):
    p = softmax_probs(logits)
    top2 = torch.topk(p, 2, dim=1).values
    return top2[:, 0] - top2[:, 1]
