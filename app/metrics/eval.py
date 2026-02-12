import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def binary_metrics(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return cm, (fpr, tpr, roc_auc), y_pred
