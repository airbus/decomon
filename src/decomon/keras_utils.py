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
        nb_missing_batch_axe_x = nb_batch_axe_y - nb_batch_axe_x
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


def add_tensors(
    x: Tensor,
    y: Tensor,
    missing_batchsize: tuple[bool, bool] = (False, False),
    diagonal: tuple[bool, bool] = (False, False),
) -> Tensor:
    """Sum tensors in a compatible way.

    Generate broadcastable versions of the tensors before summing them,
    depending on 2 characteristics:

    - missing batchsize?
    - diagonal representation?

    We only have to modify the tensors if a characteristic differ between them.
    More precisely:
    - missing batchsize: we add a batch axis (of dimension 1)
    - diagonal representation: we make a full representation of the tensor

    Args:
        x:
        y:
        missing_batchsize:
        diagonal:

    Returns:

    """
    # get broadcastable version of the tensors
    x_broadcastable, y_broadcastable = _convert_to_broacastable_tensors(
        x=x, y=y, missing_batchsize=missing_batchsize, diagonal=diagonal
    )
    # operate on broadcastable tensors
    return x_broadcastable + y_broadcastable


def _convert_to_broacastable_tensors(
    x: Tensor,
    y: Tensor,
    missing_batchsize: tuple[bool, bool],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor]:
    x_broadcastable = x
    y_broadcastable = y
    x_full_shape = x.shape
    y_full_shape = y.shape
    if missing_batchsize == (True, False):
        batchsize = y_full_shape[0]
        x_full_shape = (batchsize,) + x_full_shape
        x_broadcastable = x_broadcastable[None]
    elif missing_batchsize == (False, True):
        batchsize = x_full_shape[0]
        y_full_shape = (batchsize,) + y_full_shape
        y_broadcastable = y_broadcastable[None]
    if diagonal == (True, False):
        x_broadcastable, x_full_shape = _convert_from_diag_to_generic(
            x_broadcastable=x_broadcastable, x_full_shape=x_full_shape, missing_batchsize=all(missing_batchsize)
        )
    elif diagonal == (False, True):
        y_broadcastable, y_full_shape = _convert_from_diag_to_generic(
            x_broadcastable=y_broadcastable, x_full_shape=y_full_shape, missing_batchsize=all(missing_batchsize)
        )
    # check shapes
    if x_full_shape != y_full_shape:
        raise ValueError(
            f"Incompatible shapes: {x.shape} and {y.shape}, "
            f"with missing_batchsize={missing_batchsize} and diagonal={diagonal}."
        )

    return x_broadcastable, y_broadcastable


def _convert_from_diag_to_generic(
    x_broadcastable: Tensor, x_full_shape: tuple[int, ...], missing_batchsize: bool = False
) -> tuple[Tensor, tuple[int, ...]]:
    if missing_batchsize:
        x_full_shape = x_full_shape + x_full_shape
        new_shape = x_broadcastable.shape + x_broadcastable.shape
        x_broadcastable = K.reshape(K.diag(K.ravel(x_broadcastable)), new_shape)
    else:
        x_full_shape = x_full_shape[:1] + x_full_shape[1:] + x_full_shape[1:]
        partial_new_shape = x_broadcastable.shape[1:] + x_broadcastable.shape[1:]
        x_broadcastable = K.concatenate(
            [
                K.reshape(K.diag(K.ravel(x_broadcastable[i])), partial_new_shape)[None]
                for i in range(len(x_broadcastable))
            ],
            axis=0,
        )

    return x_broadcastable, x_full_shape


def is_a_merge_layer(layer: Layer) -> bool:
    return hasattr(layer, "_merge_function")


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
