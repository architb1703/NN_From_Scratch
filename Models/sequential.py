import torch, torch.nn as nn
from Layers.layer import Layer

class Sequential_Model:
    def __init__(self, layers, loss_fn):
        self.layers = layers
        self.loss_fn = loss_fn
        self.y_grads = None

    def forward_pass(self, X):
        for layer in self.layers:
            X = layer(X)

        return X

    def loss_calc(self, targets, preds):
        [loss, self.y_grads] = self.loss_fn(targets, preds)
        return loss

    def backward_pass(self):
        grads = self.y_grads
        
        for layer in self.layers[::-1]:
            grads = layer.grad_calc(grads)

    def step(self, lr_rate):
        for layer in self.layers[::-1]:
            if isinstance(layer, Layer):
                layer.step(lr_rate)