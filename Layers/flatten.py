from Layers.layer import Layer
import torch, torch.nn as nn

class Flatten(Layer):
    def __init__(self):
        self.orig_shape = None

    def __call__(self, X):
        self.orig_shape = X.shape
        return X.view(-1,1)

    def grad_calc(self, grads):
        return grads.view(self.orig_shape)

    def step(self, *args):
        pass