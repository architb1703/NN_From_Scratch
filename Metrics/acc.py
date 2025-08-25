import torch, torch.nn as nn

def accuracy(targets, preds):
    assert(targets.shape == preds.shape)
    y_preds = (preds>0.5).to(targets.dtype)
    acc = torch.sum(y_preds==targets)/targets.shape[1]
    return acc