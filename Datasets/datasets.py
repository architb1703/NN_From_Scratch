import numpy as np
import torch

def stratified_split(X, y, val_split=0.2, seed=None):
    y_uniq = np.unique(y)
    train_idx, val_idx = [], []
    if(seed!=None):
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    for cl in y_uniq:
        idxs = np.where(y==cl)[0]
        t_idxs = rng.choice(idxs, int((1-val_split)*len(idxs)), replace=False)
        train_idx.append(t_idxs)
        val_idx.append(np.setdiff1d(idxs, t_idxs))

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)

    return X[train_idx,:,:], y[train_idx], X[val_idx,:,:], y[val_idx]

class DataLoader:
    def __init__(self, X_train, y_train, batch_size=32, shuffle=False, device=None):
        if(device == None):
            device = X_train.device

        idxs = torch.randperm(X_train.shape[0])

        self.X_train = X_train[idxs]
        self.y_train = y_train[:,idxs]
        self.batch_size = batch_size

    def __iter__(self):
        return self.generator()

    def generator(self):
        for i in range(0, self.X_train.shape[0], self.batch_size):
            yield [self.X_train[i:i+self.batch_size], self.y_train[:,i:i+self.batch_size]]