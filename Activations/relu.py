import torch, torch.nn as nn

class relu:
    def __init__(self):
        self.activations = None

    def __call__(self, X):
        self.activations = torch.maximum(torch.zeros(X.shape).to(X.device),X)
        return self.activations

    def grad_calc(self, grads):
        return grads * (self.activations>0).to(grads.dtype)