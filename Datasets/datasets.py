import struct
import numpy as np

class MNIST_DataLoader:
    def __init__(self, rel_path="./Datasets/MNIST/"):
        self.X_train_path = rel_path + "train-images.idx3-ubyte"
        self.y_train_path = rel_path + "train-labels.idx1-ubyte"
        self.X_test_path = rel_path + "t10k-images.idx3-ubyte"
        self.y_test_path = rel_path + "t10k-labels.idx1-ubyte"
        
    def read_mnist_labels(self, filepath):
        with open(filepath, 'rb') as f:
            magic, num_images = struct.unpack('>II', f.read(8))
            # Ensure magic number is correct (0x00000803 for images)
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images)
        return images

    def read_mnist_images(self, filepath):
        with open(filepath, 'rb') as f:
            magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
            # Ensure magic number is correct (0x00000803 for images)
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        return images

    def load_data(self):
        self.X_train_data = self.read_mnist_images(self.X_train_path)
        self.y_train_data = self.read_mnist_labels(self.y_train_path)
        self.X_test_data = self.read_mnist_images(self.X_test_path)
        self.y_test_data = self.read_mnist_labels(self.y_test_path)

        return self.X_train_data, self.y_train_data, self.X_test_data, self.y_test_data

def stratified_split(X, y, val_split=0.2):
    y_uniq = np.unique(y)
    train_idx, val_idx = [], []
    
    for cl in y_uniq:
        idxs = np.where(y==cl)[0]
        t_idxs = np.random.choice(idxs, int((1-val_split)*len(idxs)), replace=False)
        train_idx.append(t_idxs)
        val_idx.append(np.setdiff1d(idxs, t_idxs))

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)

    return X[train_idx,:,:], y[train_idx], X[val_idx,:,:], y[val_idx]

