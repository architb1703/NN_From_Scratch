from math import ceil, sqrt
import torch, torch.nn as nn
from Layers.layer import Layer

class Conv2DLayer(Layer):
    def __init__(self, window_size=(3,3), stride=(1,1), channels_in=32, channels_out=32):
        self.window_size = window_size
        self.stride = stride
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.previous_hidden_statechannels_out = channels_out
        
        self.weights = torch.normal(0,sqrt(2/(channels_in*window_size[0]*window_size[1])),(channels_out,channels_in,window_size[0],window_size[1]))
        self.bias = torch.zeros((channels_out,1))
        
        self.previous_hidden_state = None
        self.w_grads = torch.zeros(self.weights.shape)
        self.b_grads = torch.zeros(self.bias.shape)

    def output_shape(self, input_shape):
        x1 = input_shape[1]-(self.window_size[0]-1)
        x1 = ceil(x1/self.stride[0])
        x2 = input_shape[2]-(self.window_size[1]-1)
        x2 = ceil(x2/self.stride[1])
        return (x1,x2)

    def __call__(self, X):
        output_shape = self.output_shape(X.shape)
        self.previous_hidden_state = X
        H = torch.zeros((self.channels_out,output_shape[0],output_shape[1]))

        for ch_out in range(H.shape[0]):
            for i in range(H.shape[1]):
                for j in range(H.shape[2]):
                    row_idx, col_idx = i*self.stride[0], j*self.stride[1]
                    mat_in = X[:,row_idx:row_idx+self.window_size[0],col_idx:col_idx+self.window_size[1]]
                    H[ch_out][i][j] = torch.sum(torch.mul(mat_in,self.weights[ch_out,:,:,:])) + self.bias[ch_out]
        return H

    def grad_calc(self, grads):
        #calculating wgrads
        for n_out in range(grads.shape[0]):
            for i in range(grads.shape[1]):
                for j in range(grads.shape[2]):
                    row_idx, col_idx = i*self.stride[0], j*self.stride[1]
                    mat_in = self.previous_hidden_state[:,row_idx:row_idx+self.window_size[0],col_idx:col_idx+self.window_size[1]]
                    self.w_grads[n_out] += mat_in*grads[n_out][i][j]
                    self.b_grads[n_out] += grads[n_out][i][j]

        self.h_grads = torch.zeros(self.previous_hidden_state.shape)
        for n_out in range(grads.shape[0]):
            for i in range(grads.shape[1]):
                for j in range(grads.shape[2]):
                    row_idx, col_idx = i*self.stride[0], j*self.stride[1]
                    self.h_grads[:,row_idx:row_idx+self.window_size[0],col_idx:col_idx+self.window_size[1]] += grads[n_out][i][j]*self.weights[n_out]
                    mat_in = self.previous_hidden_state[:,row_idx:row_idx+self.window_size[0],col_idx:col_idx+self.window_size[1]]
        
        return self.h_grads

    def step(self, lr_rate):
        self.weights -= lr_rate*self.w_grads
        self.bias -= lr_rate*self.b_grads 