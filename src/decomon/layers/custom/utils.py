import keras
import keras.ops as K
import numpy as np

from typing import Tuple, List
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

    dtype32: str = "float32"
    dtype: str = K.dtype(lower)

    o_value: Tensor = K.cast(1.0, dtype)
    z_value: Tensor = K.cast(0.0, dtype)

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

    corners_collapse: Tensor = mask * l_reshaped + (o_value - mask) * (
        u_reshaped + index_collapse
    )  # ?????? right shape but right values ?
    # add the corners containing all the upper bounds
    corners_collapse = K.concatenate(
        [corners_collapse, u_reshaped + index_collapse], axis=-1
    )  # (batch, shape_prev, n_dim, shape_after, n_dim+1)
    corners: Tensor = K.concatenate([corners_, u_reshaped], axis=-1)  # (batch, shape_prev, n_dim, shape_after, n_dim+1)

    corners_pred: Tensor = K.max(corners, axis=axis_)  # (batch, shape_prev, shape_after, n_dim+1)

    # include bias in corners
    bias_corner: Tensor = o_value + K.sum(
        z_value * corners, axis_, keepdims=True
    )  # (batch, shape_prev, 1, shape_after, n_dim+1)
    corners_collapse = K.concatenate(
        [corners_collapse, bias_corner], axis=axis_
    )  # (batch, shape_prev, n_dim+1, shape_after, n_dim+1)

    dimensions: array.array = np.arange(
        len(corners.shape)
    )  # K.solve require that the matrix dimension is on the last two axis (-2, -1): A tensor of shape (..., M, M) representing the coefficients matrix. In our case M = n_dim+1

    dim_permutation: array.array = np.concatenate([dimensions[:axis_], dimensions[axis_ + 1 :], [dimensions[axis_]]])

    corners_collapse = K.transpose(
        corners_collapse, tuple(dim_permutation)
    )  # (batch, shape_prev, shape_after, n_dim+1, n_dim+1)
    # solve works only for float32
    if dtype != dtype32:
        corners_collapse = K.cast(corners_collapse, dtype32)
        corners_pred = K.cast(corners_pred, dtype32)

    # a: A tensor of shape (..., M, M) representing the coefficients matrix.
    # b: A tensor of shape (..., M) or (..., M, N) represeting the right-hand side or "dependent variable" matrix.
    #  PYTORCH_ENABLE_MPS_FALLBACK=1
    """
    The operator 'aten::_linalg_solve_ex.result' is not currently implemented for the MPS device. 
    If you want this op to be added in priority during the prototype phase of this feature, 
    please comment on https://github.com/pytorch/pytorch/issues/77764. 
    As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` 
    to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS
    """
    w_hull: Tensor
    w_hull = K.solve(a=corners_collapse, b=corners_pred)  # (batch, shape_prev, shape_after, n_dim+1)

    """
    try:
        w_hull = K.solve(a=corners_collapse, b=corners_pred)  # (batch, shape_prev, shape_after, n_dim+1)
    except:
        # move everything to cpu
        corners_collapse_cpu: array.array = corners_collapse.to("cpu").numpy()
        corners_pred_cpu: array.array = corners_pred.to("cpu").numpy()
        w_hull_cpu: array.array = np.linalg.solve(a=corners_collapse_cpu, b=corners_pred_cpu)
        w_hull = keras.Variable(w_hull_cpu, trainable=False)
    """
    
    if dtype != dtype32:
        w_hull = K.cast(w_hull, dtype=dtype)

    # we need to split w_hull into weights and bias components
    w_u: Tensor
    b_u: Tensor

    w_u, b_u = K.split(
        w_hull, [n_dim], axis=-1
    )  # w_u : (None, shape_prev, shape_after, n_dim), b_u: (batch, shape_prev, shape_after, 1)
    b_u = K.reshape(b_u, [-1] + shape_prev + shape_after)  # b_u (batch, shape_prev, shape_after) == lower.shape

    dim_permutation = np.concatenate(
        [dimensions[:axis_], [len(lower.shape) - 1], [e - 1 for e in dimensions[axis_ + 1 : -1]]]
    ).astype(
        "int"
    )  # to check
    w_u = K.transpose(w_u, tuple(dim_permutation))

    # due to numerical error in keras.ops.solve we need to assess that w_u, b_u
    # is correct on the set of corners, else we will add the error inside the bias

    error: Tensor = K.maximum(
        z_value,
        K.cast(corners_pred, dtype=dtype) - (K.sum(K.expand_dims(w_u, -1) * corners, axis_) + K.expand_dims(b_u, -1)),
    )
    b_u = b_u+ K.max(error, -1)

    if keepdims:
        b_u = K.expand_dims(b_u, axis_)

    return [w_u, b_u]


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

    dtype32: str = "float32"
    dtype: str = K.dtype(lower)

    o_value: Tensor = K.cast(1.0, dtype)
    z_value: Tensor = K.cast(0.0, dtype)

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

    # detect collapsed dimensions: lower[i]==upper[i]
    index_collapse = K.sign(lower - upper) + o_value  # 1 iff lower[i]==upper[i]
    # index_collapse:Tensor = K.clip(2*K.sign(lower - upper) - o_value, z_value, o_value) #1 iff lower[i]==upper[i]

    # consider V_slope uniquely
    score = upper + lower - index_collapse * K.max(upper)
    criterion = K.expand_dims(K.max(score, axis), axis)
    mask = o_value + K.sign(score - criterion)

    # if there is several maximum, mask.sum()>1, use the mean
    denum_coeff = K.maximum(K.sum(mask, axis, keepdims=True), o_value)
    mask /= denum_coeff

    bias = z_value * K.max(upper, axis=axis_, keepdims=keepdims)

    return mask, bias
