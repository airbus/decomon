# Decomon: Automatic Certified Perturbation Analysis of Neural Networks

<div align="center">
    <img src="_static/banner.jpg" width="55%" alt="decomon" align="center" />
</div>

`decomon` is a library that allows the derivation of upper and lower bounds
for the predictions of a Keras neural network with perturbed inputs.
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

Since we rely on [Keras 3](https://keras.io/keras_3/) which allows in particular Tensorflow *and* Pytorch backends,
we are opening up the possibility for a new community
to formally assess the robustness of their networks, without worrying about the technicality of
the implementation. In this way, we hope to promote the formal certification of neural networks
into safety critical systems.


```{toctree}
---
maxdepth: 2
caption: Contents
---
install
getting_started
tutorials
api/modules
contribute
Github  <https://github.com/airbus/decomon>
```
