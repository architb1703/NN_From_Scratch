from math import sqrt
import torch, torch.nn as nn
from Layers.layer import Layer

class LinearLayer(Layer):
    def __init__(self, features_in, features_out):
        self.weights = torch.normal(0,sqrt(2/features_in),(features_out, features_in))
        self.bias = torch.zeros((features_out,1))
        self.prev_hidden_state = None

        self.w_grads = None
        self.b_grads = None

    def __call__(self, X):
        self.prev_hidden_state = X
        hidden_state = self.weights@X + self.bias
        return hidden_state

    def grad_calc(self, grads):
        self.w_grads = grads@(self.prev_hidden_state.T)
        self.b_grads = torch.sum(grads, dim=1, keepdim=True)

        self.h_grads = self.weights.T @ grads
        return self.h_grads

    def step(self, lr_rate):
        self.weights -= lr_rate*self.w_grads
        self.bias -= lr_rate*self.b_grads

    def to(self, device):
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)
        self.device = device