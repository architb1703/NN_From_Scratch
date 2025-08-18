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