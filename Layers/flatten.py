from Layers.layer import Layer
import torch, torch.nn as nn

class Flatten(Layer):
    def __init__(self):
        self.orig_shape = None

    def __call__(self, X):
        self.orig_shape = X.shape
        return X.view(self.orig_shape[0],-1).T

    def grad_calc(self, grads):
        return grads.T.view(self.orig_shape)

    def step(self, *args):
        pass

    def to(self, device):
        pass