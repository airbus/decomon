from typing import List, Union

import keras.ops as K  # type:ignore
import numpy as np  # type:ignore
from keras.layers import Layer  # type:ignore

from decomon.types import Tensor  # type:ignore


def combine_affine(
    layer,
    w_in,
    b_in,
    model_input_shape_wo_batchsize,
    layer_input_shape_wo_batchsize,
    layer_output_shape_wo_batchsize,
    layer_has_multiple_outputs,
):
    # apply layer on w_in, b_in
    b_out = layer(b_in)
    w_in_ = K.reshape([-1] + layer_input_shape_wo_batchsize)(w_in)
    w_out_ = layer(w_in_)
    if layer_has_multiple_outputs:
        w_out = [
            K.reshape([-1] + model_input_shape_wo_batchsize + layer_output_shape_wo_batchsize)(w_out_i)
            for w_out_i in w_out_
        ]
    else:
        w_out = K.reshape([-1] + model_input_shape_wo_batchsize + layer_output_shape_wo_batchsize)(w_out_)

    return w_out, b_out


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

    if diagonal:
        w = layer(K.ones([1] + input_shape_wo_batch))[0] - bias
    else:
        N: int = K.prod(input_shape_wo_batch)
        w: Tensor = K.reshape(K.eye(N), [-1] + input_shape_wo_batch)
        # apply the layer on w
        w = layer(w) - bias[None]  # (N, output_shape_wo_batch)

        w = K.reshape(w, input_shape_wo_batch + output_shape_wo_batch)

    return w, bias


def apply_backward_layer(
    output_affine_bounds: list[Tensor],
    layer_backward: Layer,
    is_output_linear: bool,
    output_shape_wo_batch: list[int],
    input_shape_wo_batch: list[int],
    has_bias: bool = True,
    layer: Union[None, Layer] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """ """
    [w_l, b_l, w_u, b_u] = output_affine_bounds

    if is_output_linear:
        w_l = w_l[None]
        w_u = w_u[None]
        b_l = b_l[None]
        b_u = b_u[None]

    n_out_shape = list(b_l.shape[1:])
    n_out_shape = n_out_shape[len(output_shape_wo_batch) :]
    n_out_shape_flat = int(np.prod(n_out_shape))
    # import pdb; pdb.set_trace()
    w_l_flat_0 = K.reshape(w_l, [-1] + output_shape_wo_batch + [n_out_shape_flat])  # (batch, output_shape, n_out_flat)
    w_u_flat_0 = K.reshape(w_u, [-1] + output_shape_wo_batch + [n_out_shape_flat])  # (batch, output_shape, n_out_flat)

    # permute dimension
    N_output_shape = len(output_shape_wo_batch)  # number of dimensions without batch size
    output_shape_index = [i + 1 for i in range(N_output_shape)]

    w_l_flat = K.transpose(
        w_l_flat_0, [0, N_output_shape + 1] + output_shape_index
    )  # (batch, n_out_flat, output_shape)
    w_u_flat = K.transpose(
        w_u_flat_0, [0, N_output_shape + 1] + output_shape_index
    )  # (batch, n_out_flat, output_shape)

    w_l_flat_ = K.reshape(w_l_flat, [-1] + output_shape_wo_batch)  # (batch*n_out_flat, output_shape)
    w_u_flat_ = K.reshape(w_u_flat, [-1] + output_shape_wo_batch)  # (batch*n_out_flat, output_shape)

    # apply backward layer
    w_l_conv = layer_backward(w_l_flat_)  # (batch*n_out_flat, input_shape)
    w_u_conv = layer_backward(w_u_flat_)  # (batch*n_out_flat, input_shape)

    # reshape to (batch, n_out_flat, input_shape)
    w_l_conv = K.reshape(w_l_conv, [-1, n_out_shape_flat] + input_shape_wo_batch)
    w_u_conv = K.reshape(w_u_conv, [-1, n_out_shape_flat] + input_shape_wo_batch)

    # permute dimensions: (batch, input_shape, n_out_flat)
    # (0, 1, 2, 3, 4) -> (0, 2, 3, 4, 1)
    input_shape_index = [0] + [i + 2 for i in range(len(input_shape_wo_batch))] + [1]
    w_l_conv = K.transpose(w_l_conv, input_shape_index)
    w_u_conv = K.transpose(w_u_conv, input_shape_index)

    # reshape to (batch, input_shape, n_out)
    w_l_conv = K.reshape(w_l_conv, [-1] + input_shape_wo_batch + n_out_shape)
    w_u_conv = K.reshape(w_u_conv, [-1] + input_shape_wo_batch + n_out_shape)

    if has_bias:
        # convert bias to an additive term
        bias = get_bias(layer)  # retrieve bias component with shape output_shape
        # w_u*bias (batch_size, output_shape, n_out_shape) * (output_shape,)
        # reshape bias

        bias_ = K.reshape(bias, [-1] + output_shape_wo_batch + [1] * len(n_out_shape))
        # axis_sum = [i + 1 for i in range(len(output_shape))]
        axis_sum = output_shape_index
        bias_conv_u = K.sum(w_u * bias_, axis_sum) + b_u  # (batch_size, n_out_shape)
        bias_conv_l = K.sum(w_l * bias_, axis_sum) + b_l  # (batch_size, n_out_shape)
    else:
        bias_conv_u = b_u  # (batch_size, n_out_shape)
        bias_conv_l = b_l  # (batch_size, n_out_shape)

    if is_output_linear:
        output = [w_l_conv[0], bias_conv_l[0], w_u_conv[0], bias_conv_u[0]]
    else:
        output = [w_l_conv, bias_conv_l, w_u_conv, bias_conv_u]

    return output
