import torch
from Layers.layer import Layer

class Dropout(Layer):
    def __init__(self, dp_rate=0.5):
        self.dp_rate = dp_rate
        self.mask = None
        self.device = "cpu"

    def __call__(self, X ):
        self.mask = torch.rand(X.shape[1:]).unsqueeze(0).to(self.device)
        self.mask = self.mask > self.dp_rate

        return X*self.mask
    
    def grad_calc(self, grads):
        return grads*self.mask

    def step(self, *args):
        return super().step(*args)

    def to(self, device):
        self.device = device