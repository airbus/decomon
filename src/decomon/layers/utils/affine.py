from decomon.types import Tensor
from keras.layers import Layer
import keras.ops as K
import numpy as np
from typing import List


def get_affine_representation_wo_bias(layer: Layer, diagonal: bool = False) -> tuple[Tensor, Tensor]:
    """
    Computes the affine representation (weights and bias) of a given linear layer that does not have a bias term.

    Args:
    Layer: Keras layer
    diagonal (bool): A flag indicating whether to return a diagonal weight matrix. Defaults to False

    Returns:
     the affine representation (w, b) such that: layer(x)= W*x
    """

    input_shape_wo_batch: List[int] = list(layer.input.shape[1:])
    N: int = np.prod(input_shape_wo_batch)
    w: Tensor = K.reshape(K.eye(N), [-1] + input_shape_wo_batch)
    # apply the layer on w
    w = layer(w)  # (N, output_shape_wo_batch)
    output_shape_wo_batch: List[int] = list(layer.output.shape[1:])
    b: Tensor = K.zeros(output_shape_wo_batch)

    if diagonal:
        w = K.reshape(K.diag(K.reshape(w, (N, N))), input_shape_wo_batch)
    else:
        w = K.reshape(w, input_shape_wo_batch + output_shape_wo_batch)

    return w, b


def get_bias(layer: Layer) -> Tensor:
    """
    Computes the bias component of a given linear layer.

    Args:
    Layer: Keras layer

    Returns:
     the affine representation b such that: layer(x)= W*x + b
    """

    input_shape_wo_batch: List[int] = list(layer.input.shape[1:])

    w_b: Tensor = K.zeros([1] + input_shape_wo_batch)
    bias: Tensor = layer(w_b)[0]  # output_shape_wo_batch

    w_b = K.expand_dims(K.zeros(input_shape_wo_batch), 0)
    bias = layer(w_b)[0]  # output_shape_wo_batch
    return bias


def get_affine_representation_with_bias(layer: Layer, diagonal: bool = False) -> tuple[Tensor, Tensor]:
    """
    Computes the affine representation (weights and bias) of a given linear layer.

    Args:
    Layer: Keras layer
    diagonal (bool): A flag indicating whether to return a diagonal weight matrix. Defaults to False

    Returns:
     the affine representation (w, b) such that: layer(x)= W*x + b
    """

    input_shape_wo_batch: List[int] = list(layer.input.shape[1:])
    output_shape_wo_batch: List[int] = list(layer.output.shape[1:])

    w_b: Tensor = K.zeros([1] + input_shape_wo_batch)
    bias: Tensor = layer(w_b)[0]  # output_shape_wo_batch

    N: int = K.prod(input_shape_wo_batch)
    w: Tensor = K.reshape(K.eye(N), [-1] + input_shape_wo_batch)
    # apply the layer on w
    w = layer(w) - bias[None]  # (N, output_shape_wo_batch)

    if diagonal:
        w = K.reshape(K.diag(K.reshape(w, (N, N))), input_shape_wo_batch)
    else:
        w = K.reshape(w, input_shape_wo_batch + output_shape_wo_batch)

    return w, bias
