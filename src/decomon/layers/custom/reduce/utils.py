import keras
import keras.ops as K
import numpy as np

from decomon.types import Tensor


def get_affine_upper_bound_max_before_reduction(
    lower: Tensor, upper: Tensor, axis: int, keepdims: bool = True
) -> tuple[Tensor, Tensor]:
    """The get_affine_upper_bound_max function computes an affine upper bound approximation for the max function applied along a specified axis of tensors.

    Args:
        lower: A tensor containing the lower bound values along which the max operation is applied.
        upper A tensor containing the upper bound values for the same operation.
        axis: The axis along which to compute the max function and over-approximation.
        keepdims: If True, keeps the reduced dimension in the result. Otherwise, the dimension is removed.

    Returns:
       w_u: The weight tensor of the affine approximation for the max function.
       b_u:The bias tensor of the affine approximation.

    NB: w_u needs to be multiplied to input then summed over the proper axis so that

        K.sum(w_u * x, axis=axis, keepdims=keepdims) + b_u >= K.max(x, axis=axis, keepdims=keepdims)

    """

    dtype: str = K.dtype(lower)

    o_value: Tensor = K.cast(1.0, dtype)

    N: int = len(lower.shape)
    axis_: int

    if axis < 0:
        axis_ = len(lower.shape) + axis
    else:
        axis_ = axis

    # permute data so that the operator is on the last dimension ... ?
    input_shape: list[int] = list(lower.shape)  # (batch, shape_prev, n_dim, shape_after)
    # get the shape of the dimension
    n_dim: int = input_shape[axis_]  # n_dim

    # expand dim/broadcast
    mask: Tensor = K.eye(n_dim)  # (n_dim, n_dim)

    mask_shape = np.ones(len(lower.shape) + 1, dtype="int")
    mask_shape[-1] = n_dim
    mask_shape[axis_] = n_dim

    mask = K.reshape(mask, tuple(mask_shape))  # (1, 1.., n_dim, 1.., n_dim)

    l_reshaped: Tensor = K.expand_dims(lower, -1)  # (batch, shape_prev, n_dim, shape_after, 1)
    u_reshaped: Tensor = K.expand_dims(upper, -1)  # (batch, shape_prev, n_dim, shape_after, 1)

    corners_: Tensor = mask * l_reshaped + (o_value - mask) * (
        u_reshaped
    )  # (batch, shape_prev, n_dim, shape_after, n_dim)

    corners_pred: Tensor = K.max(corners_, axis=axis_)  # (batch, shape_prev, shape_after, n_dim)

    max_pred = K.max(u_reshaped, axis=axis_)  # (batch, shape_prev, shape_after, 1)

    w_u_ = (max_pred - corners_pred) / K.maximum(K.sum(u_reshaped - corners_, axis_), keras.backend.epsilon())
    # (batch, shape_prev, shape_after, n_dim)

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
    if keepdims:
        corners_pred = K.expand_dims(corners_pred, axis_)
    error = corners_pred - K.sum(K.expand_dims(w_u, -1) * corners_, axis_, keepdims=keepdims) - K.expand_dims(b_u, -1)
    b_u = b_u + K.max(error, -1)

    # set w_l=0 and b_l = lower whenever lower=upper
    mask_collapse = K.max(K.sign(upper - lower), axis=axis, keepdims=keepdims)  # (None, shape_before, shape_after)
    # mask_collapse[i] = 0 if upper == lower
    if keepdims:
        w_u = mask_collapse * w_u
    else:
        w_u = K.expand_dims(mask_collapse, axis) * w_u
    b_u = mask_collapse * b_u + (1 - mask_collapse) * K.max(lower, axis=axis, keepdims=keepdims)

    return (w_u, b_u)


def get_batch_multi_dot_repr_for_axis_reduce_weights(w: Tensor, axis: int, keepdims: bool) -> Tensor:
    """Transform weights to be compatible with batch_multi_dot

    We transform w into ww such that

        batch_multi_dot(x, ww ) = K.sum(w*x, axis=axis, keepdims=keepdims)

    where x and w share same shapes and include batch dimension.

    Args:
        w:
        axis:
        keepdims:

    Returns:

    """
    # compute positive value for axis
    if axis < 0:
        axis_ = len(w.shape) + axis
    else:
        axis_ = axis

    if axis_ < 1:
        raise NotImplementedError()

    input_shape_wo_batch = tuple(w.shape[1:])
    batchsize = w.shape[0]
    if keepdims:
        reduced_axis_shape: tuple[int, ...] = (1,)
    else:
        reduced_axis_shape = tuple()
    output_shape_wo_batch = input_shape_wo_batch[: axis_ - 1] + reduced_axis_shape + input_shape_wo_batch[axis_:]
    N = int(np.prod(output_shape_wo_batch))
    identity_reshaped = K.reshape(K.eye(N), (1,) + output_shape_wo_batch + output_shape_wo_batch)
    if not keepdims:
        identity_reshaped = K.expand_dims(identity_reshaped, axis_)
    return identity_reshaped * K.reshape(w, (batchsize,) + input_shape_wo_batch + (1,) * len(output_shape_wo_batch))


def max_prime(inputs: Tensor, axis: int) -> Tensor:
    # preprocessing: need to overcome max nb of dimension for argmax (7 with tensorflow) => reshape
    if axis < 0:
        axis_ = len(inputs.shape) + axis
    else:
        axis_ = axis
    oldshape = inputs.shape
    newshape = (int(np.prod(inputs.shape[:axis_])), inputs.shape[axis_], int(np.prod(inputs.shape[axis_ + 1 :])))
    inputs_reshaped = K.reshape(inputs, newshape=newshape)
    axis_reshaped = 1

    # max_prime: one-hot encoding of argmax
    indices = K.argmax(inputs_reshaped, axis_reshaped)
    dim_i = inputs_reshaped.shape[axis_reshaped]
    output_reshaped = K.one_hot(indices, dim_i, axis=axis_reshaped)

    # postprocessing: reshape back
    output = K.reshape(output_reshaped, newshape=oldshape)
    return output


def get_affine_lower_bound_max_before_reduction(
    lower: Tensor, upper: Tensor, axis: int, keepdims: bool = True
) -> tuple[Tensor, Tensor]:
    """The get_affine_lower_bound_max function computes an affine lower bound approximation for the max function applied along a specified axis of tensors.

    Args:
        lower: A tensor containing the lower bound values along which the max operation is applied.
        upper A tensor containing the upper bound values for the same operation.
        axis: The axis along which to compute the max function and over-approximation.
        keepdims: If True, keeps the reduced dimension in the result. Otherwise, the dimension is removed.

    Returns:
       w_l: The weight tensor of the affine approximation for the max function.
       b_l:The bias tensor of the affine approximation.

    NB: w_L needs to be multiplied to input then summed over the proper axis so that

        K.sum(w_l * x, axis=axis, keepdims=keepdims) + b_l <= K.max(x, axis=axis, keepdims=keepdims)

    """
    # compute a solution for lower bound
    w_l_lower = max_prime(lower, axis=axis)  # (None, shape_before, axis, shape_after)
    b_l_lower = K.max(lower, axis=axis, keepdims=keepdims) - K.sum(
        w_l_lower * lower, axis, keepdims=keepdims
    )  # (None, shape_before, shape_after)
    # compute a solution for upper bound
    w_l_upper = max_prime(upper, axis=axis)  # (None, shape_before, axis, shape_after)
    b_l_upper = K.max(upper, axis=axis, keepdims=keepdims) - K.sum(
        w_l_upper * upper, axis, keepdims=keepdims
    )  # (None, shape_before, shape_after)

    # take the one that maximize the integral in the range [lower, upper]
    score_lower = b_l_lower * K.sum(upper - lower, axis, keepdims=keepdims) + K.sum(
        w_l_lower * (upper - lower), axis, keepdims=keepdims
    )
    score_upper = b_l_upper * K.sum(upper - lower, axis, keepdims=keepdims) + K.sum(
        w_l_upper * (upper - lower), axis, keepdims=keepdims
    )

    if keepdims:
        w_l = K.where(score_lower >= score_upper, w_l_lower, w_l_upper)
    else:
        w_l = K.where(K.expand_dims(score_lower, axis) >= K.expand_dims(score_upper, axis=axis), w_l_lower, w_l_upper)
    b_l = K.where(score_lower >= score_upper, b_l_lower, b_l_upper)

    # set w_l=0 and b_l = lower whenever lower=upper
    mask_collapse = K.max(K.sign(upper - lower), axis=axis, keepdims=keepdims)  # (None, shape_before, shape_after)
    # mask_collapse[i] = 0 if upper == lower
    if keepdims:
        w_l = mask_collapse * w_l
    else:
        w_l = K.expand_dims(mask_collapse, axis) * w_l
    b_l = mask_collapse * b_l + (1 - mask_collapse) * K.max(lower, axis=axis, keepdims=keepdims)

    return (w_l, b_l)
