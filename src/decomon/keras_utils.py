from typing import Any, List

import keras
import keras.ops as K
import numpy as np
from keras.layers import Layer

from decomon.types import BackendTensor, Tensor

BACKEND_TENSORFLOW = "tensorflow"
BACKEND_PYTORCH = "torch"
BACKEND_NUMPY = "numpy"
BACKEND_JAX = "jax"


class LinalgSolve(keras.Operation):
    """Keras operation mimicking tensorflow.linalg.solve()."""

    def compute_output_spec(self, matrix: keras.KerasTensor, rhs: keras.KerasTensor) -> keras.KerasTensor:
        rhs_shape = rhs.shape
        rhs_dtype = getattr(rhs, "dtype", type(rhs))
        rhs_sparse = getattr(rhs, "sparse", False)
        return keras.KerasTensor(
            shape=rhs_shape,
            dtype=rhs_dtype,
            sparse=rhs_sparse,
        )

    def call(self, matrix: BackendTensor, rhs: BackendTensor) -> BackendTensor:
        backend = keras.config.backend()
        if backend == BACKEND_TENSORFLOW:
            import tensorflow as tf

            return tf.linalg.solve(matrix=matrix, rhs=rhs)
        elif backend == BACKEND_PYTORCH:
            import torch

            return torch.linalg.solve(A=matrix, B=rhs)
        else:
            raise NotImplementedError(f"linalg_solve() not yet implemented for backend {backend}.")


class BatchedIdentityLike(keras.Operation):
    """Keras Operation creating an identity tensor with shape (including batch_size) based on input.

    The output shape is tuple(x.shape) + (x.shape[-1],), the tensor being the identity
    along the 2 last dimensions.

    """

    def call(self, x: BackendTensor) -> Tensor:
        input_shape = x.shape
        identity_tensor = K.identity(input_shape[-1], dtype=x.dtype)
        n_repeat = int(np.prod(input_shape[:-1]))
        return K.reshape(K.repeat(identity_tensor[None], n_repeat, axis=0), tuple(input_shape) + (-1,))

    def compute_output_spec(self, x: Tensor) -> keras.KerasTensor:
        x_shape = x.shape
        x_type = getattr(x, "dtype", type(x))
        x_sparse = getattr(x, "sparse", False)
        return keras.KerasTensor(
            shape=tuple(x_shape) + (x_shape[-1],),
            dtype=x_type,
            sparse=x_sparse,
        )


class BatchedDiagLike(keras.Operation):
    """Keras Operation transforming last dimension into a diagonal tensor.

    The output shape is tuple(x.shape) + (x.shape[-1],).
    When fixing all but 2 last dimensions, the output tensor is a square tensor
    whose main diagonal is the input tensor with same first dimensions fixed, and 0 elsewhere.

    This is a replacement for tensorflow.linalg.diag().

    """

    def call(self, x: BackendTensor) -> Tensor:
        return K.concatenate([K.diag(K.ravel(w_part))[None] for w_part in K.split(x, len(x), axis=0)], axis=0)

    def compute_output_spec(self, x: Tensor) -> keras.KerasTensor:
        x_shape = x.shape
        x_type = getattr(x, "dtype", type(x))
        x_sparse = getattr(x, "sparse", False)
        return keras.KerasTensor(
            shape=tuple(x_shape) + (x_shape[-1],),
            dtype=x_type,
            sparse=x_sparse,
        )


def is_symbolic_tensor(x: Tensor) -> bool:
    """Check whether the tensor is symbolic or not.

    Works even during backend calls made by layers without actual compute_output_shape().
    In this case, x is not KerasTensor anymore but a backend Tensor with None in its shape.

    """
    return None in x.shape


def get_weight_index(layer: Layer, weight: keras.Variable) -> int:
    """Get weight index among layer tracked weights

    Args:
        layer: layer we are looking
        weight: weight supposed to be part of tracked weights by the layer

    Returns:
        the index of the weight in `layer.weights` list

    Raises:
        IndexError: if `weight` is not part of `layer.weights`

    """
    indexes = [i for i, w in enumerate(layer.weights) if w is weight]
    try:
        return indexes[0]
    except IndexError:
        raise IndexError(f"The weight {weight} is not tracked by the layer {layer}.")


def get_weight_index_from_name(layer: Layer, weight_name: str) -> int:
    """Get weight index among layer tracked weights

    Args:
        layer: layer we are looking
        weight_name: name of the weight supposed to be part of tracked weights by the layer

    Returns:
        the index of the weight in `layer.weights` list

    Raises:
        AttributeError: if `weight_name` is not the name of an attribute of `layer`
        IndexError: if the corresponding layer attribute is not part of `layer.weights`

    """
    weight = getattr(layer, weight_name)
    try:
        return get_weight_index(layer=layer, weight=weight)
    except IndexError:
        raise IndexError(f"The weight {weight_name} is not tracked by the layer {layer}.")


def reset_layer(new_layer: Layer, original_layer: Layer, weight_names: List[str]) -> None:
    """Reset some weights of a layer by using the weights of another layer.

    Args:
        new_layer: the decomon layer whose weights will be updated
        original_layer: the layer used to update the weights
        weight_names: the names of the weights to update

    Returns:

    """
    if not original_layer.built:
        raise ValueError(f"the layer {original_layer.name} has not been built yet")
    if not new_layer.built:
        raise ValueError(f"the layer {new_layer.name} has not been built yet")
    else:
        new_params = new_layer.get_weights()
        original_params = original_layer.get_weights()
        for weight_name in weight_names:
            new_params[get_weight_index_from_name(new_layer, weight_name)] = original_params[
                get_weight_index_from_name(original_layer, weight_name)
            ]
        new_layer.set_weights(new_params)


def reset_layer_all_weights(new_layer: Layer, original_layer: Layer) -> None:
    """Reset all the weights of a layer by using the weights of another layer.

    Args:
        new_layer: the decomon layer whose weights will be updated
        original_layer: the layer used to update the weights

    Returns:

    """
    reset_layer(new_layer=new_layer, original_layer=original_layer, weight_names=[w.name for w in new_layer.weights])


def share_layer_all_weights(
    original_layer: Layer,
    new_layer: Layer,
) -> None:
    """Share all the weights of an already built layer to another unbuilt layer.

    Args:
        original_layer: the layer used to share the weights
        new_layer: the new layer which will be buit and will share the weights of the original layer

    Returns:

    """
    share_weights_and_build(
        new_layer=new_layer, original_layer=original_layer, weight_names=[w.name for w in original_layer.weights]
    )


def share_weights_and_build(original_layer: Layer, new_layer: Layer, weight_names: List[str]) -> None:
    """Share the weights specidifed by names of an already built layer to another unbuilt layer.

    We assume that each weight is also an original_laer's attribute whose name is the weight name.

    Args:
        original_layer: the layer used to share the weights
        new_layer: the new layer which will be buit and will share the weights of the original layer
        weight_names: names of the weights to share

    Returns:

    """
    # Check the original_layer is built and the new_layer is not built
    if not original_layer.built:
        raise ValueError("The original layer must already be built for sharing its weights.")
    if new_layer.built:
        raise ValueError("The new layer must not be built to get the weights of the original layer")
    # Check that input exists really (ie that the layer has already been called on a symbolic KerasTensor
    inp = original_layer.input  # will raise a ValueError if not existing

    # store the weights as a new_layer variable before build (ie before the lock)
    for w_name in weight_names:
        w = getattr(original_layer, w_name)
        setattr(new_layer, w_name, w)

    # build the layer
    new_layer(inp)
    # overwrite the newly generated weights and untrack them
    for w_name in weight_names:
        w = getattr(original_layer, w_name)
        w_to_drop = getattr(new_layer, w_name)
        setattr(new_layer, w_name, w)
        # untrack the not used anymore kernel
        new_layer._tracker.untrack(w_to_drop)


def check_if_single_shape(shape: Any) -> bool:
    """

    Args:
        input_shape:

    Returns:

    """
    if isinstance(shape, list) and shape and isinstance(shape[0], (int, type(None))):
        return True

    if not isinstance(shape, (list, tuple, dict)):
        shape = tuple(shape)

    return isinstance(shape, tuple) and len(shape) > 0 and isinstance(shape[0], (int, type(None)))
