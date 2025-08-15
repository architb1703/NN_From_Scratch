import torch, torch.nn as nn

class sigmoid:
    def __init__(self):
        self.activations = None

    def __call__(self, X):
        self.activations = 1/(1+torch.exp(-1*X))
        return self.activations

    def grad_calc(self, grads):
        return grads * (self.activations*(1-self.activations))