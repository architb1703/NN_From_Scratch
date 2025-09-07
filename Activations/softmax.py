import torch

class softmax:
    def __init__(self):
        self.activations = None
    
    def __call__(self, X):
        self.activations = torch.exp(X)
        self.activations = self.activations/torch.sum(self.activations,dim=-1).unsqueeze(-1)

        return self.activations

    def grad_calc(self, grads):
        softmax_weights = torch.diag_embed(self.activations) - (self.activations.unsqueeze(2) @ self.activations.unsqueeze(1))
        softmax_grads = softmax_weights@grads.unsqueeze(2)
        return softmax_grads.squeeze(2)