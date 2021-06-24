# DECOMON: Automatic Monotonic Decomposition Coupled with Linear Relaxation for Certified Robustness of Neural Networks under Perturbations

## Introduction

**What is Decomon?** `Decomon` is a library for automatically deriving upper and lower bounds 
for a Tensorflow/Keras neural network
with perturbed inputs. These bounds are represented as affine functions
with respect to some variable under perturbation.
Since previous works that tackled certified robustness with backward propagation still relied on forward
upper and lower bounds, in `Decomon`we studied various ways to tighten forward upper and
lower bounds, while remaining backpropagation compatible
 thanks to symbolic optimization.

Our algorithm improves existing forward linear relaxation algorithms for general
Keras based Neural Networks without manual derivation. Our implementation is also automatically
**differentiable**. So far we support interval bound propagation, forward mode perturbation, backward mode perturbation and their hybrid approaches as well. 
`Decomon is compatible with a wider range of perturbation:
boxes, $L_{\inf, 1, 2}$ norms or general convex sets described by their vertices.

Decomon appeared to us as a complementary tool to existing libraries for the certification of neural networks. 
Since we rely on Tensorflow and not Pytorch, we are opening up the possibility for a new community
to formally assess the robustness of their networks, without worrying about the technicality of
the implementation. In this way, we hope to promote the formal certification of neural networks 
into safety critical systems.


If you encounter any problems with this library, feel free create an issue or pull request. We
welcome contributions in any form from anyone. Please make sure beforehand that your contribution
is aligned with [Black](https://github.com/psf/black) format.

## Installation

Python 3.7+ and Tensorflow 2.4 is required. We recommend to manage your python version using 
[asdf](https://asdf-vm.com/#/core-manage-asdf)

Before you run any examples, please install `Decomon` first:

```
git clone https://github.com/airbus/decomon
cd decomon
poetry build
```

This library is still under heavy development.

## Quick Start

First define your Keras Neural Network and convert it into its `Decomon` version
thanks to the `convert` method. Then you can call respectively `get_upper_box`
and `get_lower_box` to obtain certified upper and lower bounds for the network's outputs
within a box domain. 

````python
# import
from decomon.models import convert
from decomon import get_upper_box, get_lower_box, get_range_box, \
                    get_range_noise

# Toy example with a Keras model:
model = Sequential([Dense(10, activation='relu', input_dim=2)])

# Create a fake box with the right shape
x_min = np.zeros(1, 2)
x_max = np.ones(1, 2)

# convert into a Decomon neural network
decomon_model = convert(model)

upper_bound = get_upper_box(decomon_model, x_min, x_max)
lower_bound = get_lower_box(decomon_model, x_min, x_max)
````

As mentioned other types of domains are possible and illustrated 
in our [tutorials](tutorials).

## Units Tests

Pytest is required to run unit tests (pip install pytest).

Make sure you are in the `decomon` root directory and run unit tests (the "-v" verbose mode is optional but gives additional details):

```
cd YOUR_LOCAL_PATH_TO_GIT_CLONED_DECOMON

poetry run pytest tests -v
```
