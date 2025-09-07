from math import ceil
import torch
from Layers.layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, window_size=(2,2), stride=(2,2)):
        self.window_size = window_size
        self.stride = stride
        self.device = "cpu"
        self.prev_hidden_state_shape = None

    def output_shape(self, X):
        x1, x2 = X.shape[-2], X.shape[-1]
        out1 = ceil((x1-(self.window_size[0]-1))/self.stride[0])
        out2 = ceil((x2-(self.window_size[1]-1))/self.stride[1])
        return out1,out2

    def __call__(self, X):
        out1,out2 = self.output_shape(X)
        N,C_in,_,_ = X.shape
        H = torch.zeros((N, C_in, out1, out2)).to(self.device)
        self.max_idx = []
        self.prev_hidden_state_shape = X.shape

        for n in range(N):
            for c in range(C_in):
                for i in range(out1):
                    for j in range(out2):
                        s1, s2 = i*self.stride[0], j*self.stride[1]
                        mat_in = X[n,c,s1:s1+self.window_size[0],s2:s2+self.window_size[1]]
                        H[n,c,i,j] = torch.max(mat_in)
                        idx = torch.argmax(mat_in)
                        self.max_idx.append([idx//self.window_size[1],idx%self.window_size[1]])
        return H

    def grad_calc(self, grads):
        new_grads = torch.zeros(self.prev_hidden_state_shape)

        N,C,I,J = grads.shape
        idx=0
        for n in range(N):
            for c in range(C):
                for i in range(I):
                    for j in range(J):
                        max_arg = self.max_idx[idx]
                        s1, s2 = i*self.stride[0], j*self.stride[1]
                        new_grads[n,c,s1+max_arg[0],s2+max_arg[1]] = grads[n,c,i,j]
                        idx += 1

        return new_grads

    def step(self, *args):
        return super().step(*args)

    def to(self, device):
        self.device = device
        