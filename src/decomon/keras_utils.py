from typing import Any, Optional

import keras
import keras.ops as K
import numpy as np
from keras.layers import Dot, Layer, Reshape

from decomon.types import BackendTensor, Tensor

BACKEND_TENSORFLOW = "tensorflow"
BACKEND_PYTORCH = "torch"
BACKEND_NUMPY = "numpy"
BACKEND_JAX = "jax"


def batch_multid_dot(
    x: Tensor,
    y: Tensor,
    nb_merging_axes: Optional[int] = None,
    missing_batchsize: tuple[bool, bool] = (False, False),
    diagonal: tuple[bool, bool] = (False, False),
) -> Tensor:
    """Dot product of tensors by batch, along multiple axes

    Hypothesis: we sum over last axes of x and first axes (skipping the batch one) of x.

    The 1-dimensional equivalent would be `batch_dot(x,y, axes=(-1, 1))`
    or `keras.layers.Dot(axes=(-1, 1))(x,y)`

    Args:
        x:
        y:
        nb_merging_axes: number of axes to be merged.
          By default, all (non-batch) axes of x,i.e.
          len(x.shape) if missing_batchsize[0] else len(x.shape) - 1
        missing_batchsize: specify if a tensor is missing the batch dimension, for x and y.
            In that case, the corresponding tensor is broadcasted accordingly.
        diagonal: specify is a tensor is only represented by its diagonal. See below for an example.

    Returns:

    For performance reasons, instead of actually repeating the tensor `batchsize` along a new first axis,
    we rather use `keras.ops.tensordot` directly on tensors without broadcasting them.

    Notes:
        - The dimensions of axes along which we perform the dot product
         (i.e. x.shape[-nb_merging_axes:] and y.shape[1:1 + nb_merging_axes] when no batch axe is missing) should match.
        - diagonal example: if x is diagonal and missing its batch axis, it means that the full tensor x is retrieve with
             x_full = K.reshape(K.diag(K.ravel(x)), x.shape + x.shape)
          With the batch axis, the above computation should be made batch element by batch element.

    """
    diag_x, diag_y = diagonal
    missing_batchsize_x, missing_batchsize_y = missing_batchsize
    nb_batch_axe_x = 0 if missing_batchsize_x else 1
    nb_batch_axe_y = 0 if missing_batchsize_y else 1
    if nb_merging_axes is None:
        nb_merging_axes = len(x.shape) - nb_batch_axe_x

    # check shapes compatibility
    x_merging_axes_shape = x.shape[-nb_merging_axes:]
    y_merging_axes_shape = y.shape[nb_batch_axe_y : nb_batch_axe_y + nb_merging_axes]
    if x_merging_axes_shape != y_merging_axes_shape:
        raise ValueError(
            "Incompatible input shapes: "
            f"Merging axes dimension should match. "
            f"Found {x_merging_axes_shape} and {y_merging_axes_shape}. "
            f"Full shapes: {x.shape} and {y.shape}."
        )

    # Special cases: diagonal entries (represented only by their diagonal)
    if diag_x and diag_y:
        # all inputs diagonal: we keep a diagonal output (with batch axis if one input has one)
        return x * y
    elif diag_x:
        # reshape to make broadcast possible
        nb_missing_batch_axe_x = 1 - nb_batch_axe_x
        nb_missing_axes_x_wo_batch = len(y.shape) - nb_batch_axe_y - len(x.shape) + nb_batch_axe_x
        new_shape = nb_missing_batch_axe_x * (1,) + x.shape + (1,) * nb_missing_axes_x_wo_batch
        return K.reshape(x, new_shape) * y
    elif diag_y:
        # reshape necessary for broadcast, only if y has a batch axis
        if not missing_batchsize_y:
            nb_missing_axes_y_wo_batch = len(x.shape) - nb_batch_axe_x - len(y.shape) + nb_batch_axe_y
            new_shape = y.shape[:1] + (1,) * nb_missing_axes_y_wo_batch + y.shape[1:]
            return x * K.reshape(y, new_shape)
        else:
            return x * y
    else:
        # switch on missing batch axe (e.g. with affine layer representation like Dense's kernel)
        if missing_batchsize_y:
            return K.tensordot(x, y, axes=nb_merging_axes)
        elif missing_batchsize_x:
            # axes along which summing
            merging_axes_x = list(range(-nb_merging_axes, 0))
            merging_axes_y = list(range(nb_batch_axe_y, nb_batch_axe_y + nb_merging_axes))
            # transposition to make to put back batch axe at the beginning
            nb_axes_after_merge = len(x.shape) + len(y.shape) - 2 * nb_merging_axes
            nb_axes_after_merge_from_x = len(x.shape) - nb_merging_axes
            transpose_indices = (
                (nb_axes_after_merge_from_x,)
                + tuple(range(nb_axes_after_merge_from_x))
                + tuple(range(nb_axes_after_merge_from_x + 1, nb_axes_after_merge))
            )
            return K.transpose(K.tensordot(x, y, axes=[merging_axes_x, merging_axes_y]), transpose_indices)
        else:
            new_x_shape = tuple(x.shape[1:-nb_merging_axes]) + (-1,)
            new_y_shape = (-1,) + tuple(y.shape[nb_merging_axes + 1 :])
            return Dot(axes=(-1, 1))([Reshape(new_x_shape)(x), Reshape(new_y_shape)(y)])


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


def is_a_merge_layer(layer: Layer) -> bool:
    return hasattr(layer, "_merge_function")


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


def reset_layer(new_layer: Layer, original_layer: Layer, weight_names: list[str]) -> None:
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


def share_weights_and_build(original_layer: Layer, new_layer: Layer, weight_names: list[str]) -> None:
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
        try:
            setattr(new_layer, w_name, w)
        except AttributeError:
            # manage hidden weights introduced for LoRA https://github.com/keras-team/keras/pull/18942
            w_name = f"_{w_name}"
            w = getattr(original_layer, w_name)
            setattr(new_layer, w_name, w)

    # build the layer
    new_layer(inp)
    # overwrite the newly generated weights and untrack them
    for w_name in weight_names:
        w = getattr(original_layer, w_name)
        w_to_drop = getattr(new_layer, w_name)
        try:
            setattr(new_layer, w_name, w)
        except AttributeError:
            # manage hidden weights introduced for LoRA https://github.com/keras-team/keras/pull/18942
            w_name = f"_{w_name}"
            w = getattr(original_layer, w_name)
            w_to_drop = getattr(new_layer, w_name)
            setattr(new_layer, w_name, w)
        # untrack the not used anymore weight
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
