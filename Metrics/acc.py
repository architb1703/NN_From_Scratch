import torch, torch.nn as nn

def accuracy(targets, preds):
    y_preds = (preds>0.5).to(targets.dtype)
    acc = torch.sum(y_preds==targets)/targets.shape[0]
    return acc