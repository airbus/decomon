from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from decomon.layers.core import ForwardMode, StaticVariables
from decomon.utils import get_lower, get_upper

# step 1: compute (x_i, y_i) such that x_i[j]=l_j if j==i else u_j
# dataset of size n+1 on which we can compute an affine bound


def get_upper_linear_hull_max(
    inputs: List[tf.Tensor],
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    convex_domain: Optional[Dict[str, Any]] = None,
    axis: int = -1,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Compute the linear hull that overapproximates max along the axis dimension

    Args:
        inputs: list of input tensors
        mode: type of Forward propagation (ibp, affine, or hybrid). Default to hybrid.
        convex_domain (optional): type of convex domain that encompass the set of perturbations. Defaults to None.
        axis (optional): Defaults to -1. See Keras offical documentation backend.max(., axis)

    Raises:
        NotImplementedError: axis <0 and axis!=-1

    Returns:
        list of output tensors. The upper linear relaxation of max(., axis) in the mode format
    """

    # attention if axis=-1 or axis=n
    dtype = inputs[-1].dtype
    dtype32 = "float32"
    mode = ForwardMode(mode)
    o_value = K.cast(1.0, dtype)
    z_value = K.cast(0.0, dtype)
    if axis == len(inputs[-1].shape):
        axis = -1
    if axis != -1 and axis < 0:
        raise NotImplementedError()  # to do

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:

        u_c, l_c = inputs[:nb_tensor]

    elif mode == ForwardMode.AFFINE:

        x, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
        u_c = get_upper(x, w_u, b_u, convex_domain=convex_domain)
        # assuming axis=-1 u_c.shape=(None, shape, n_dim)
        l_c = get_lower(x, w_l, b_l, convex_domain=convex_domain)

    elif mode == ForwardMode.HYBRID:

        x, u_c_h, w_u, b_u, l_c_h, w_l, b_l = inputs[:nb_tensor]
        u_c = K.minimum(u_c_h, get_upper(x, w_u, b_u, convex_domain=convex_domain))
        l_c = K.maximum(l_c_h, get_lower(x, w_l, b_l, convex_domain=convex_domain))

    # get the shape of the dimension
    n_dim = K.int_shape(inputs[-1])[axis]

    # expand dim/broadcast
    mask = tf.linalg.diag(tf.ones((n_dim), dtype=dtype))  # (n_dim, n_dim)
    mask_shape = np.ones(len(u_c.shape) + 1)
    mask_shape[-1] = n_dim
    if axis != -1:
        mask_shape[axis] = n_dim
    else:
        mask_shape[-2] = n_dim
    mask = K.reshape(mask, mask_shape)  # (1, shape, n_dim, n_dim)

    l_c_ = K.expand_dims(l_c, -1)  # (1, shape, n_dim, 1)
    u_c_ = K.expand_dims(u_c, -1)  # (1, shape, n_dim, 1)

    corners = mask * l_c_ + (o_value - mask) * u_c_  # (1, shape, n_dim, n_dim)
    # add the corners containing all the upper bounds
    corners = K.concatenate([corners, u_c_], axis=-1)  # (None, shape, n_dim, n_dim+1)

    if axis != -1:
        corners_pred = K.max(corners, axis=axis)  # (None, shape_, n_dim+1)
    else:
        corners_pred = K.max(corners, axis=-2)  # (None, shape_, n_dim+1)

    # include bias in corners
    if axis != -1:
        bias_corner = o_value + tf.math.reduce_sum(z_value * corners, axis, keepdims=True)
        corners_ = K.concatenate([corners, bias_corner], axis=axis)  # (None, shape_, n_dim+1, n_dim+1)
    else:
        bias_corner = o_value + tf.math.reduce_sum(z_value * corners, -2, keepdims=True)
        corners_ = K.concatenate([corners, bias_corner], axis=-2)

    dimensions = np.arange(len(corners_.shape))
    if axis != -1:
        dim_ = np.concatenate([dimensions[:axis], dimensions[axis + 1 :], [dimensions[axis]]])
    else:
        dim_ = np.concatenate([dimensions[:-2], dimensions[-1:], [dimensions[-2]]])

    corners_ = K.permute_dimensions(corners_, dim_)  # do it properly for different axis !!!
    # corners_ = K.permute_dimensions(corners, ())
    # tf.linalg.solve works only for float32
    if dtype != dtype32:
        corners_ = K.cast(corners_, dtype32)
        corners_pred = K.cast(corners_pred, dtype32)
    w_hull = tf.linalg.solve(matrix=corners_, rhs=K.expand_dims(corners_pred, -1))  # (None, shape_, n_dim+1, 1)
    if dtype != dtype32:
        w_hull = K.cast(w_hull, dtype=dtype)

    shape_prev = np.prod(inputs[-1].shape[1:axis]).astype("int")
    if axis == -1:
        shape_after = 1
    else:
        shape_after = np.prod(inputs[-1].shape[axis + 1 :]).astype("int")
    # (-1, shape_prev, axis, shape_after, n_dim+1)
    shape_ = np.prod(w_hull.shape[1:-2])
    w_hull_flat = K.reshape(w_hull, (-1, shape_, n_dim + 1))
    w_u = K.reshape(w_hull_flat[:, :, :-1], (-1, shape_prev, shape_after, n_dim))  # (-1, shape_, n_dim)
    w_u = K.reshape(K.permute_dimensions(w_u, (0, 1, 3, 2)), [-1] + list(inputs[-1].shape[1:]))
    # reshape w_u
    shape_max = K.max(inputs[-1], axis).shape[1:]
    b_u = K.reshape(w_hull_flat[:, :, -1], [-1] + list(shape_max))  # (-1, shape_)

    return [w_u, b_u]


def get_lower_linear_hull_max(
    inputs: List[tf.Tensor],
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    convex_domain: Optional[Dict[str, Any]] = None,
    axis: int = -1,
    finetune_lower: Optional[tf.Tensor] = None,
    **kwargs: Any,
) -> List[tf.Tensor]:
    """Compute the linear hull that overapproximates max along the axis dimension

    Args:
        inputs: list of input tensors
        mode: type of Forward propagation (ibp, affine, or hybrid). Default to hybrid.
        convex_domain (optional): type of convex domain that encompass the set of perturbations. Defaults to None.
        axis (optional): Defaults to -1. See Keras offical documentation backend.max(., axis)
        finetune_lower: If not None, should be a constant tensor used to fine tune the lower relaxation.

    Raises:
        NotImplementedError: axis <0 and axis!=-1

    Returns:
        list of output tensors. The lower linear relaxation of max(., axis) in the mode format
    """

    dtype = inputs[-1].dtype
    mode = ForwardMode(mode)
    o_value = K.cast(1.0, dtype)
    z_value = K.cast(0.0, dtype)
    if axis == len(inputs[-1].shape):
        axis = -1
    if axis != -1 and axis < 0:
        raise NotImplementedError()  # to do

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == ForwardMode.IBP:

        u_c, l_c = inputs[:nb_tensor]

    elif mode == ForwardMode.AFFINE:

        x, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
        u_c = get_upper(x, w_u, b_u, convex_domain=convex_domain)
        # assuming axis=-1 u_c.shape=(None, shape, n_dim)
        l_c = get_lower(x, w_l, b_l, convex_domain=convex_domain)

    elif mode == ForwardMode.HYBRID:

        x, u_c_h, w_u, b_u, l_c_h, w_l, b_l = inputs[:nb_tensor]
        u_c = K.minimum(u_c_h, get_upper(x, w_u, b_u, convex_domain=convex_domain))
        l_c = K.maximum(l_c_h, get_lower(x, w_l, b_l, convex_domain=convex_domain))

    V = u_c + l_c
    M = -u_c + K.expand_dims(K.max(l_c, axis), axis)
    # M>=0 means that the associated coordinate cannot be an argmax
    # we can use M to add a penalty to the index that cannot be an argmax

    V += K.expand_dims(K.max(K.abs(V), axis), axis) * (K.sign(M) + o_value)

    w_l = K.cast(tf.one_hot(K.argmin(V, axis), depth=V.shape[axis], axis=axis), dtype)
    b_l = K.sum(z_value * w_l, axis)
    # without finetuning: consider a one hot vector with value one at index=argmin(V)

    if finetune_lower is not None:
        alpha = finetune_lower[None]
        y = alpha * u_c + (o_value - alpha) * l_c
        w_l_alpha = K.cast(tf.one_hot(K.argmax(y, axis), depth=V.shape[axis], axis=axis), dtype)
        w_l = (o_value - alpha) * w_l + alpha * w_l_alpha

    return [w_l, b_l]