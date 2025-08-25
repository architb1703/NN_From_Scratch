# NN_From_Scratch - Training neural networks without torch.nn

My goal with this repo is to implement neural network concepts from scratch, without using any of the specialized neural network libraries like torch.nn. This, I feel, is really important to think more deeply about the functions we so quickly use when training models otherwise. I started from a simple linear model, and the goal is to be able to reproduce some simple papers/architectures using just this library.

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
