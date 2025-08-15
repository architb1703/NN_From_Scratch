import torch, torch.nn as nn

def log_loss(targets, preds):
    m = preds.shape[1]
    loss_arr = targets*(torch.log(preds+1e-8)) + (1-targets)*(torch.log(1-preds+1e-8))
    loss = -1*torch.sum(loss_arr)/m

    grads = (-1/m)*(targets/(preds+1e-8) - (1-targets)/(1-preds+1e-8))
    
    return [loss, grads]