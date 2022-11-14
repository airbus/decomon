<div align="center">
    <img src="docs/assets/banner.jpg" width="55%" alt="Decomon" align="center" />
</div>
<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.6, 3.7, 3.8-efefef">
    </a>
    <a href="https://github.com/airbus/decomon/actions/workflows/python-lints.yml">
        <img alt="PyLint" src="https://github.com/airbus/decomon/actions/workflows/python-lints.yml/badge.svg">
    </a>
    <a href="https://github.com/airbus/decomon/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/airbus/decomon/actions/workflows/python-tests.yml/badge.svg">
    </a>
     <a href="https://github.com/airbus/decomon/actions/workflows/python-publish.yml">
        <img alt="Pypi" src="https://github.com/airbus/decomon/actions/workflows/python-publish.yml/badge.svg">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>
<br>

# DecoMon: Automatic Certified Perturbation Analysis of Neural Networks

## Introduction

**What is DecoMon?** `DecoMon` is a library that allows the derivation of upper and lower bounds
for the predictions of a Tensorflow/Keras neural network with perturbed inputs.
In the current release, these bounds are represented as affine functions with respect to some variable under perturbation.

Previous works that tackled certified robustness with backward propagation relied on forward upper and lower bounds. In `DecoMon`,
we explored various ways to tighten forward upper and lower bounds, while remaining backpropagation-compatible
 thanks to symbolic optimization.

Our algorithm improves existing forward linear relaxation algorithms for general Keras-based neural networks
without manual derivation. Our implementation is also automatically **differentiable**.
So far we support interval bound propagation, forward mode perturbation, backward mode perturbation as well as hybrid approaches.

`DecoMon` is compatible with a wider range of perturbation: boxes, $L_{\inf, 1, 2}$ norms or general
convex sets described by their vertices.

We believe that DecoMon is a complementary tool to existing libraries for the certification of neural networks.

Since we rely on Tensorflow and not Pytorch, we are opening up the possibility for a new community
to formally assess the robustness of their networks, without worrying about the technicality of
the implementation. In this way, we hope to promote the formal certification of neural networks
into safety critical systems.

If you encounter any problems with this library, feel free to create an issue or a pull request. We
welcome contributions in any form from anyone. Please make sure beforehand that your contribution
is aligned with [Black](https://github.com/psf/black) format.

## Installation

Python 3.7+ and Tensorflow 2.4 are required. We recommend that you manage your python version using
[asdf](https://asdf-vm.com/#/core-manage-asdf)

Please install DecoMon first before running any examples following

```
git clone https://github.com/airbus/decomon
cd decomon
poetry build
```

This library is still under heavy development.

## Quick Start

First define your Keras Neural Network and convert it into its `DecoMon` version
using the `convert` method. You can then call `get_upper_box` and `get_lower_box` to
respectively obtain certified upper and lower bounds for the network's outputs
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

# convert into a DecoMon neural network
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
