<div align="center">
    <img src="https://raw.githubusercontent.com/airbus/decomon/main/docs/source/_static/banner.jpg" width="55%" alt="decomon" align="center" />
</div>
<br>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-%E2%89%A53.7-efefef">
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

# Decomon: Automatic Certified Perturbation Analysis of Neural Networks

## Introduction

**What is decomon?** `decomon` is a library that allows the derivation of upper and lower bounds
for the predictions of a Tensorflow/Keras neural network with perturbed inputs.
In the current release, these bounds are represented as affine functions with respect to some variable under perturbation.

Previous works that tackled certified robustness with backward propagation relied on forward upper and lower bounds. In `decomon`,
we explored various ways to tighten forward upper and lower bounds, while remaining backpropagation-compatible
 thanks to symbolic optimization.

Our algorithm improves existing forward linear relaxation algorithms for general Keras-based neural networks
without manual derivation. Our implementation is also automatically **differentiable**.
So far we support interval bound propagation, forward mode perturbation, backward mode perturbation as well as hybrid approaches.

`decomon` is compatible with a wider range of perturbation: boxes, $L_{\inf, 1, 2}$ norms or general
convex sets described by their vertices.

We believe that decomon is a complementary tool to existing libraries for the certification of neural networks.

Since we rely on Tensorflow and not Pytorch, we are opening up the possibility for a new community
to formally assess the robustness of their networks, without worrying about the technicality of
the implementation. In this way, we hope to promote the formal certification of neural networks
into safety critical systems.


## Installation

Quick version:
```shell
pip install decomon
```
For more details, see the [online documentation](https://airbus.github.io/decomon/main/install).

## Quick start

You can see how to get certified lower and upper bounds for a basic Keras neural network in the
[Getting started section](https://airbus.github.io/decomon/main/getting_started) of the online documentation.


## Documentation

The latest documentation is available [online](https://airbus.github.io/decomon).

## Examples

Some educational notebooks are available in `tutorials/` folder.
Links to launch them online with [colab](https://colab.research.google.com/) or [binder](https://mybinder.org/) are provided in the
[Tutorials section](https://airbus.github.io/decomon/main/tutorials) of the online documentation.

## Contributing

We welcome any contribution. See more about how to contribute in the [online documentation](https://airbus.github.io/decomon/main/contribute).
