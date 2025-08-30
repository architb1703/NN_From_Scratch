# NN_From_Scratch

## Setup

To run the notebooks in this project, or to create your own, you first need to install the project package. From the root directory of the project, run the following command:

```bash
pip install -e .
```

This will install the project in "editable" mode, meaning any changes you make to the source files will be immediately available.

After installation, you can import classes and functions from the project's modules in your notebooks. For example:

```python
from Layers.linear import LinearLayer

# Now you can use the LinearLayer class
layer = LinearLayer(10, 5)
```

## Implemented Machine Learning Concepts

Here's what has been implemented so far:

### Neural Network Layers
- **Linear/Dense Layer** (`Layers/linear.py`): Fully connected layer with weight initialization using He initialization, forward pass, backward pass with gradient computation, and parameter updates
- **Convolutional Layer** (`Layers/conv.py`): 2D convolution operation with configurable window size, stride, and channels, including gradient computation for backpropagation and parameter updates
- **Flatten Layer** (`Layers/flatten.py`): Reshapes multi-dimensional tensors to 2D for transition between convolutional and dense layers

### Activation Functions
- **ReLU** (`Activations/relu.py`): Rectified Linear Unit activation with gradient computation for backpropagation
- **Sigmoid** (`Activations/sigmoid.py`): Sigmoid activation function with gradient computation for backpropagation

### Loss Functions
- **Binary Cross-Entropy Loss** (`Loss_Func/log_loss.py`): Logarithmic loss for binary classification with gradient computation

### Evaluation Metrics
- **Accuracy** (`Metrics/acc.py`): Binary classification accuracy metric
- **F1 Score** (`Metrics/f1.py`): F1 score, precision and recall calculation for binary classification

### Model Architecture
- **Sequential Model** (`Models/sequential.py`): Class for linking the layers of the model in a sequential manner, with support for forward-pass, backward-pass etc.

### Data Handling
- **MNIST DataLoader** (`Datasets/mnist_dataloader.py`): Custom data loader for MNIST dataset with binary file parsing
- **Stratified Split** (`Datasets/datasets.py`): Data splitting utility that maintains class distribution in train/validation sets
- **Batch DataLoader** (`Datasets/datasets.py`): Mini-batch data loading with shuffling capabilities

### Additional Features
- **Device Management**: Support for moving computations between CPU and GPU (/"mps" for Mac)