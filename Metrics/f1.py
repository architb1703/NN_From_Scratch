import torch

def binary_precision(targets, preds):
    true_positive = torch.sum(preds[targets==1])
    total_positive_preds = torch.sum(preds==1)
    return true_positive/(total_positive_preds)

def binary_recall(targets, preds):
    true_positive = torch.sum(preds[targets==1])
    total_positive = torch.sum(targets==1)
    return true_positive/(total_positive)

def f1_score(targets, preds):
    assert(targets.shape == preds.shape)
    preds = (preds>0.5).to(targets.dtype)

    true_positive = torch.sum(preds[targets==1])
    false_positive = torch.sum(preds[targets==0])
    false_negative = torch.sum(targets==1)-true_positive

    return (2*true_positive)/(2*true_positive + false_negative + false_positive)