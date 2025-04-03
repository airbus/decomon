from typing import List, Tuple

import keras  # type:ignore
import keras.ops as K  # type:ignore
import numpy as np  # type:ignore

from decomon.types import Tensor


def get_affine_upper_bound_max(lower: Tensor, upper: Tensor, axis: int, keepdims: bool = True) -> Tuple[Tensor, Tensor]:
    """The get_affine_upper_bound_max function computes an affine upper bound approximation for the max function applied along a specified axis of tensors.

    Args:
        lower: A tensor containing the lower bound values along which the max operation is applied.
        upper A tensor containing the upper bound values for the same operation.
        axis: The axis along which to compute the max function and over-approximation.
        keepdims: If True, keeps the reduced dimension in the result. Otherwise, the dimension is removed.

    Returns:
       w_u: The weight tensor of the affine approximation for the max function.
       b_u:The bias tensor of the affine approximation.

    Warning: due to multiple usage, the shape of w_u and b_u are the following:
    if lower if of shape (batch, shape_prev, axis_dim, shape_after) then w_u's shape is (batch, shape_prev, axis_dim, shape_after) and b_u's shape is (batch, shape_prev, shape_after)

    """

    dtype: str = K.dtype(lower)

    o_value: Tensor = K.cast(1.0, dtype)

    N: int = len(lower.shape)
    axis_: int

    if axis < 0:
        axis_ = len(lower.shape) + axis
    else:
        axis_ = axis

    shape_prev: List[int] = list(lower.shape[1:axis_])
    shape_after: List[int]

    if axis_ == N - 1:
        shape_after = []
    else:
        shape_after = list(lower.shape[axis_ + 1 :])

    # permute data so that the operator is on the last dimension ... ?
    input_shape: List[int] = list(lower.shape)  # (batch, shape_prev, n_dim, shape_after)
    # get the shape of the dimension
    n_dim: int = input_shape[axis_]  # n_dim

    # expand dim/broadcast
    mask: Tensor = K.eye(n_dim)  # (n_dim, n_dim)

    mask_shape: array.array = np.ones(len(lower.shape) + 1, dtype="int")
    mask_shape[-1] = n_dim
    mask_shape[axis_] = n_dim

    mask: Tensor = K.reshape(mask, tuple(mask_shape))  # (1, 1.., n_dim, 1.., n_dim)

    l_reshaped: Tensor = K.expand_dims(lower, -1)  # (batch, shape_prev, n_dim, shape_after, 1)
    u_reshaped: Tensor = K.expand_dims(upper, -1)  # (batch, shape_prev, n_dim, shape_after, 1)

    # detect collapsed dimensions: lower[i]==upper[i]
    index_collapse: Tensor = K.sign(l_reshaped - u_reshaped) + o_value  # 1 iff lower[i]==upper[i]
    # index_collapse:Tensor = K.clip(2*K.sign(l_reshaped - u_reshaped) - o_value, z_value, o_value) #1 iff lower[i]==upper[i]

    corners_: Tensor = mask * l_reshaped + (o_value - mask) * (
        u_reshaped
    )  # (batch, shape_prev, n_dim, shape_after, n_dim)

    corners_pred: Tensor = K.max(corners_, axis=axis_, keepdims=keepdims)  # (batch, shape_prev, shape_after, n_dim)

    max_pred = K.max(u_reshaped, axis=axis_, keepdims=keepdims)  # (batch, shape_prev, shape_after, 1)

    w_u_ = (max_pred - corners_pred) / K.maximum(
        K.sum(u_reshaped - corners_, axis_, keepdims=keepdims), keras.backend.epsilon()
    )
    # (batch, shape_prev, shape_after, n_dim)
    # denum = K.maximum(upper - lower, keras.backend.epsilon())
    # w_u_ = (max_pred - corners_pred)

    # to do: permute axis
    n_dim = len(w_u_.shape)
    perm_axis = [i for i in range(n_dim)]

    perm_axis = perm_axis[:axis_] + perm_axis[-1:] + perm_axis[axis_:-1]
    w_u = K.transpose(w_u_, perm_axis)
    # w_u_ = K.transpose(w_u_, perm_axis)/denum

    b_u = -K.sum(w_u * upper, axis_, keepdims=keepdims) + K.max(
        upper, axis_, keepdims=keepdims
    )  # (batch, shape_prev, shape_after)

    # due to numerical error we need to assess that w_u, b_u
    # is correct on the set of corners, else we will add the error inside the bias

    error = corners_pred - K.sum(K.expand_dims(w_u, -1) * corners_, axis_, keepdims=keepdims) - K.expand_dims(b_u, -1)
    b_u = b_u + K.max(error, -1)

    # set w_l=0 and b_l = lower whenever lower=upper
    mask_collapse = K.max(K.sign(upper - lower), axis=axis)  # (None, shape_before, shape_after)
    # mask_collapse[i] = 0 if upper == lower
    w_u = K.expand_dims(mask_collapse, axis) * w_u
    b_u = mask_collapse * b_u + (1 - mask_collapse) * K.max(lower, axis=axis)

    return [w_u, b_u]


def max_prime(inputs, axis: int):
    indices = K.argmax(inputs, axis)
    dim_i = inputs.shape[axis]
    output = K.one_hot(indices, dim_i, axis=axis)
    return output


def get_affine_lower_bound_max(lower: Tensor, upper: Tensor, axis: int, keepdims: bool = True) -> Tuple[Tensor, Tensor]:
    """The get_affine_lower_bound_max function computes an affine lower bound approximation for the max function applied along a specified axis of tensors.

    Args:
        lower: A tensor containing the lower bound values along which the max operation is applied.
        upper A tensor containing the upper bound values for the same operation.
        axis: The axis along which to compute the max function and over-approximation.
        keepdims: If True, keeps the reduced dimension in the result. Otherwise, the dimension is removed.

    Returns:
       w_l: The weight tensor of the affine approximation for the max function.
       b_l:The bias tensor of the affine approximation.

    Warning: due to multiple usage, the shape of w_l and b_l are the following:
    if lower if of shape (batch, shape_prev, axis_dim, shape_after) then w_l's shape is (batch, shape_prev, axis_dim, shape_after) and b_l's shape is (batch, shape_prev, shape_after)

    """

    # compute a solution for lower bound
    w_l_lower = max_prime(lower, axis=axis)  # (None, shape_before, axis, shape_after)
    b_l_lower = K.max(lower, axis=axis) - K.sum(w_l_lower * lower, axis)  # (None, shape_before, shape_after)
    # compute a solution for upper bound
    w_l_upper = max_prime(upper, axis=axis)  # (None, shape_before, axis, shape_after)
    b_l_upper = K.max(upper, axis=axis) - K.sum(w_l_upper * upper, axis)  # (None, shape_before, shape_after)

    # take the one that maximize the integral in the range [lower, upper]
    score_lower = b_l_lower * K.sum(upper - lower, axis) + K.sum(w_l_lower * (upper - lower), axis)
    score_upper = b_l_upper * K.sum(upper - lower, axis) + K.sum(w_l_upper * (upper - lower), axis)

    w_l = K.where(K.expand_dims(score_lower, axis) >= K.expand_dims(score_upper, axis=axis), w_l_lower, w_l_upper)
    b_l = K.where(score_lower >= score_upper, b_l_lower, b_l_upper)

    # set w_l=0 and b_l = lower whenever lower=upper
    mask_collapse = K.max(K.sign(upper - lower), axis=axis)  # (None, shape_before, shape_after)
    # mask_collapse[i] = 0 if upper == lower
    w_l = K.expand_dims(mask_collapse, axis) * w_l
    b_l = mask_collapse * b_l + (1 - mask_collapse) * K.max(lower, axis=axis)

    if keepdims:
        raise NotImplementedError()

    return [w_l, b_l]
