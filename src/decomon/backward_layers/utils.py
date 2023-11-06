from typing import Any, List, Optional, Tuple, Union

import keras.ops as K
import numpy as np
from keras.config import floatx
from keras.layers import Flatten

from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    get_affine,
    get_ibp,
    get_lower_box,
    get_upper_box,
)
from decomon.keras_utils import BatchedIdentityLike
from decomon.layers.utils import sort
from decomon.types import Tensor
from decomon.utils import maximum, minus, relu_, subtract


def backward_add(
    inputs_0: List[Tensor],
    inputs_1: List[Tensor],
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
) -> List[List[Tensor]]:
    """Backward  LiRPA of inputs_0+inputs_1

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    op_flat = Flatten(dtype=floatx())  # pas terrible  a revoir
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_0
    )
    x_1, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1, h_1, g_1 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_1
    )

    u_c_0 = op_flat(u_c_0)
    u_c_1 = op_flat(u_c_1)
    l_c_0 = op_flat(l_c_0)
    l_c_1 = op_flat(l_c_1)

    upper_0 = get_upper_box(l_c_0, u_c_0, w_u_out, b_u_out)
    upper_1 = get_upper_box(l_c_1, u_c_1, w_u_out, b_u_out)
    lower_0 = get_lower_box(l_c_0, u_c_0, w_l_out, b_l_out)
    lower_1 = get_lower_box(l_c_1, u_c_1, w_l_out, b_l_out)

    w_u_out_0 = w_u_out
    b_u_out_0 = upper_1
    w_l_out_0 = w_l_out
    b_l_out_0 = lower_1

    w_u_out_1 = w_u_out
    b_u_out_1 = upper_0
    w_l_out_1 = w_l_out
    b_l_out_1 = lower_0

    return [[w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0], [w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1]]


def backward_linear_prod(
    x_0: Tensor,
    bounds_x: List[Tensor],
    back_bounds: List[Tensor],
    perturbation_domain: Optional[PerturbationDomain] = None,
) -> List[Tensor]:
    """Backward  LiRPA of a subroutine prod

    Args:
        bounds_x
        back_bounds

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    z_value = K.cast(0.0, x_0.dtype)
    o_value = K.cast(1.0, x_0.dtype)

    w_u_i, b_u_i, w_l_i, b_l_i = bounds_x
    w_u, b_u, w_l, b_l = back_bounds

    if len(w_u_i.shape) > 3:
        n_dim = w_u_i.shape[1]
        w_u_i = K.reshape(w_u_i, (-1, n_dim, n_dim))
        w_l_i = K.reshape(w_l_i, (-1, n_dim, n_dim))
        b_u_i = K.reshape(b_u_i, (-1, n_dim))
        b_l_i = K.reshape(b_l_i, (-1, n_dim))

    x_max = perturbation_domain.get_upper(x_0, w_u_i - w_l_i, b_u_i - b_l_i)
    mask_b = o_value - K.sign(x_max)
    mask_a = o_value - mask_b

    w_u_i_expanded = K.expand_dims(K.expand_dims(w_u_i, 1), -1)
    w_l_i_expanded = K.expand_dims(K.expand_dims(w_l_i, 1), -1)
    b_u_i_expanded = K.expand_dims(K.expand_dims(b_u_i, 1), -1)
    b_l_i_expanded = K.expand_dims(K.expand_dims(b_l_i, 1), -1)
    mask_a = K.expand_dims(K.expand_dims(mask_a, 1), -1)
    mask_b = K.expand_dims(K.expand_dims(mask_b, 1), -1)

    w_u_pos = K.maximum(w_u, z_value)
    w_u_neg = K.minimum(w_u, z_value)
    w_l_pos = K.maximum(w_l, z_value)
    w_l_neg = K.minimum(w_l, z_value)

    w_u_pos_expanded = K.expand_dims(w_u_pos, 2)
    w_u_neg_expanded = K.expand_dims(w_u_neg, 2)
    w_l_pos_expanded = K.expand_dims(w_l_pos, 2)
    w_l_neg_expanded = K.expand_dims(w_l_neg, 2)
    mask_a_expanded = K.expand_dims(mask_a, 2)
    mask_b_expanded = K.expand_dims(mask_b, 2)

    w_u_out = K.sum(
        mask_a_expanded * (w_u_pos_expanded * w_u_i_expanded + w_u_neg_expanded * w_l_i_expanded), 3
    ) + K.sum(K.expand_dims(w_u, 2) * mask_b_expanded * w_u_i_expanded, 3)
    w_l_out = K.sum(
        mask_a_expanded * (w_l_pos_expanded * w_l_i_expanded + w_l_neg_expanded * w_u_i_expanded), 3
    ) + K.sum(K.expand_dims(w_l, 2) * mask_b_expanded * w_l_i_expanded, 3)

    b_u_out = (
        K.sum(mask_a * (w_u_pos * b_u_i_expanded + w_u_neg * b_l_i_expanded), 2)
        + K.sum(mask_b * (w_u * b_u_i_expanded), 2)
        + b_u
    )
    b_l_out = (
        K.sum(mask_a * (w_l_pos * b_l_i_expanded + w_l_neg * b_u_i_expanded), 2)
        + K.sum(mask_b * (w_l * b_l_i_expanded), 2)
        + b_l
    )

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def backward_maximum(
    inputs_0: List[Tensor],
    inputs_1: List[Tensor],
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
    **kwargs: Any,
) -> List[List[Tensor]]:
    """Backward  LiRPA of maximum(inputs_0, inputs_1)

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    input_step_a_0 = subtract(
        inputs_0, inputs_1, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode
    )
    input_step_0 = relu_(
        input_step_a_0, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, **kwargs
    )
    _, bounds_1 = backward_add(
        input_step_0,
        inputs_1,
        w_u_out,
        b_u_out,
        w_l_out,
        b_l_out,
        perturbation_domain=perturbation_domain,
        mode=mode,
        dc_decomp=dc_decomp,
    )

    input_step_a_1 = subtract(
        inputs_1, inputs_0, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode
    )
    input_step_1 = relu_(
        input_step_a_1, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode, **kwargs
    )
    _, bounds_0 = backward_add(
        input_step_1,
        inputs_0,
        w_u_out,
        b_u_out,
        w_l_out,
        b_l_out,
        perturbation_domain=perturbation_domain,
        mode=mode,
        dc_decomp=dc_decomp,
    )

    return [bounds_0, bounds_1]


# convex hull of the maximum between two functions
def backward_max_(
    inputs: List[Tensor],
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    axis: int = -1,
    dc_decomp: bool = False,
    **kwargs: Any,
) -> List[Tensor]:
    """Backward  LiRPA of max

    Args:
        inputs: list of tensors
        dc_decomp: boolean that indicates
        grad_bounds: boolean that indicates whether
        perturbation_domain: the type of perturbation domain
        axis: axis to perform the maximum
    whether we return a difference of convex decomposition of our layer
    we propagate upper and lower bounds on the values of the gradient

    Returns:
        max operation  along an axis
    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    affine = get_affine(mode)
    ibp = get_ibp(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs, tight=False, compute_ibp_from_affine=False
    )
    dtype = x.dtype
    input_shape = inputs_outputs_spec.get_kerasinputshape(inputs)
    max_dim = input_shape[axis]
    if max_dim is None:
        raise ValueError(f"Dimension {axis} corresponding to `axis` cannot be None")
    empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)
    empty_tensor_list = [empty_tensor] * max_dim
    z_value = K.cast(0.0, dtype=dtype)

    # do some transpose so that the last axis is also at the end
    if ibp:
        u_c_list = K.split(u_c, max_dim, axis)
        l_c_list = K.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + z_value * (u_c_list[0])
        l_c_tmp = l_c_list[0] + z_value * (l_c_list[0])
    else:
        u_c_list, l_c_list = empty_tensor_list, empty_tensor_list
        u_c_tmp, l_c_tmp = empty_tensor, empty_tensor

    if affine:
        b_u_list = K.split(b_u, max_dim, axis)
        b_l_list = K.split(b_l, max_dim, axis)
        b_u_tmp = b_u_list[0] + z_value * (b_u_list[0])
        b_l_tmp = b_l_list[0] + z_value * (b_l_list[0])

        if axis == -1:
            w_u_list = K.split(w_u, max_dim, axis)
            w_l_list = K.split(w_l, max_dim, axis)
        else:
            w_u_list = K.split(w_u, max_dim, axis + 1)
            w_l_list = K.split(w_l, max_dim, axis + 1)
        w_u_tmp = w_u_list[0] + z_value * (w_u_list[0])
        w_l_tmp = w_l_list[0] + z_value * (w_l_list[0])
    else:
        b_u_list, b_l_list, w_u_list, w_l_list = (
            empty_tensor_list,
            empty_tensor_list,
            empty_tensor_list,
            empty_tensor_list,
        )
        b_u_tmp, b_l_tmp, w_u_tmp, w_l_tmp = empty_tensor, empty_tensor, empty_tensor, empty_tensor

    if dc_decomp:
        raise NotImplementedError()
    else:
        h_list, g_list = empty_tensor_list, empty_tensor_list
        h_tmp, g_tmp = empty_tensor, empty_tensor

    outputs = []
    output_tmp = inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
        [x, u_c_tmp, w_u_tmp, b_u_tmp, l_c_tmp, w_l_tmp, b_l_tmp, h_tmp, g_tmp]
    )
    for i in range(1, max_dim):
        output_i = inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i], h_list[i], g_list[i]]
        )
        outputs.append([[elem for elem in output_tmp], output_i])
        output_tmp = maximum(
            output_tmp, output_i, dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain
        )

    outputs = outputs[::-1]
    bounds = []
    if len(outputs) > 0:
        for input_0, input_1 in outputs:
            bounds_0, bounds_1 = backward_maximum(
                input_0, input_1, w_u_out, b_u_out, w_l_out, b_l_out, mode=mode, dc_decomp=dc_decomp, **kwargs
            )
            bounds.append(bounds_1)
            w_u_out, b_u_out, w_l_out, b_l_out = bounds_0
        bounds.append(bounds_0)
        bounds = bounds[::-1]

    if axis < 0:
        w_u_out = K.concatenate([b[0] for b in bounds], axis - 1)
        w_l_out = K.concatenate([b[2] for b in bounds], axis - 1)
        b_u_out = K.sum(K.concatenate([K.expand_dims(b[1], axis - 1) for b in bounds], axis - 1), axis - 1)
        b_l_out = K.sum(K.concatenate([K.expand_dims(b[3], axis - 1) for b in bounds], axis - 1), axis - 1)
    else:
        w_u_out = K.concatenate([b[0] for b in bounds], axis)
        w_l_out = K.concatenate([b[2] for b in bounds], axis)
        b_u_out = K.sum(K.concatenate([K.expand_dims(b[1], axis) for b in bounds], axis), axis)
        b_l_out = K.sum(K.concatenate([K.expand_dims(b[3], axis) for b in bounds], axis), axis)

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def backward_minimum(
    inputs_0: List[Tensor],
    inputs_1: List[Tensor],
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
    **kwargs: Any,
) -> List[List[Tensor]]:
    """Backward  LiRPA of minimum(inputs_0, inputs_1)

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    w_u_out, b_u_out, w_l_out, b_l_out = backward_minus(w_u_out, b_u_out, w_l_out, b_l_out)
    bounds_0, bounds_1 = backward_maximum(
        inputs_0,
        inputs_1,
        w_u_out,
        b_u_out,
        w_l_out,
        b_l_out,
        perturbation_domain=perturbation_domain,
        mode=mode,
        dc_decomp=dc_decomp,
        **kwargs,
    )

    w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0 = bounds_0
    w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1 = bounds_1

    bounds_0 = backward_minus(w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0)
    bounds_1 = backward_minus(w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1)

    return [bounds_0, bounds_1]


def backward_minus(
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
) -> List[Tensor]:
    """Backward  LiRPA of -x

    Args:
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """
    return [-w_l_out, -b_l_out, -w_u_out, -b_u_out]


def backward_scale(
    scale_factor: float,
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
) -> List[Tensor]:
    """Backward  LiRPA of scale_factor*x

    Args:
        scale_factor
        w_u_out
        b_u_out
        w_l_out
        b_l_out

    Returns:

    """

    if scale_factor >= 0:
        output = [scale_factor * w_u_out, b_u_out, scale_factor * w_l_out, b_l_out]
    else:
        output = [scale_factor * w_l_out, b_l_out, scale_factor * w_u_out, b_u_out]

    return output


def backward_subtract(
    inputs_0: List[Tensor],
    inputs_1: List[Tensor],
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
) -> List[List[Tensor]]:
    """Backward  LiRPA of inputs_0 - inputs_1

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    inputs_1 = minus(inputs_1, mode=mode, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain)
    bounds_0, bounds_1 = backward_add(
        inputs_0,
        inputs_1,
        w_u_out,
        b_u_out,
        w_l_out,
        b_l_out,
        perturbation_domain=perturbation_domain,
        mode=mode,
        dc_decomp=dc_decomp,
    )

    bounds_1 = [-bounds_1[0], bounds_1[1], -bounds_1[2], bounds_1[3]]
    return [bounds_0, bounds_1]


def backward_multiply(
    inputs_0: List[Tensor],
    inputs_1: List[Tensor],
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
) -> List[List[Tensor]]:
    """Backward  LiRPA of element-wise multiply inputs_0*inputs_1

    Args:
        inputs_0
        inputs_1
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        perturbation_domain
        mode

    Returns:

    """

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_0
    )
    x_1, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1, h_1, g_1 = inputs_outputs_spec.get_fullinputs_from_inputsformode(
        inputs_1
    )

    z_value = K.cast(0.0, u_c_0.dtype)

    n = int(np.prod(u_c_0.shape[1:]))
    n_shape = [-1, n]
    # broadcast dimensions if needed
    n_out = len(w_u_out.shape[1:])
    for _ in range(n_out):
        n_shape += [1]
    a_u_0 = K.reshape(u_c_1, n_shape)
    a_u_1 = K.reshape(u_c_0, n_shape)

    b_u_0 = K.reshape((K.maximum(l_c_0, z_value) * u_c_1 + K.minimum(l_c_0, z_value) * l_c_1 - u_c_1 * l_c_0), n_shape)
    b_u_1 = K.reshape((K.maximum(l_c_1, z_value) * u_c_0 + K.minimum(l_c_1, z_value) * l_c_0 - u_c_0 * l_c_1), n_shape)

    a_l_0 = K.reshape(l_c_1, n_shape)
    a_l_1 = K.reshape(l_c_0, n_shape)
    b_l_0 = K.reshape((K.maximum(l_c_0, z_value) * l_c_1 + K.minimum(l_c_0, z_value) * u_c_1 - l_c_1 * l_c_0), n_shape)
    b_l_1 = K.reshape((K.maximum(l_c_1, z_value) * l_c_0 + K.minimum(l_c_1, z_value) * u_c_0 - l_c_0 * l_c_1), n_shape)

    # upper
    w_u_out_max = K.maximum(w_u_out, z_value)
    w_u_out_min = K.minimum(w_u_out, z_value)

    w_u_out_0 = w_u_out_max * a_u_0 + w_u_out_min * a_l_0
    w_u_out_1 = w_u_out_max * a_u_1 + w_u_out_min * a_l_1

    b_u_out_0 = K.sum(w_u_out_max * b_u_0, 1) + K.sum(w_u_out_min * b_l_0, 1) + b_u_out
    b_u_out_1 = K.sum(w_u_out_max * b_u_1, 1) + K.sum(w_u_out_min * b_l_1, 1) + b_u_out

    # lower
    w_l_out_max = K.maximum(w_l_out, z_value)
    w_l_out_min = K.minimum(w_l_out, z_value)

    w_l_out_0 = w_l_out_max * a_l_0 + w_l_out_min * a_u_0
    w_l_out_1 = w_l_out_max * a_l_1 + w_l_out_min * a_u_1

    b_l_out_0 = K.sum(w_l_out_max * b_l_0, 1) + K.sum(w_l_out_min * b_u_0, 1) + b_l_out
    b_l_out_1 = K.sum(w_l_out_max * b_l_1, 1) + K.sum(w_l_out_min * b_u_1, 1) + b_l_out

    bounds_0 = [w_u_out_0, b_u_out_0, w_l_out_0, b_l_out_0]
    bounds_1 = [w_u_out_1, b_u_out_1, w_l_out_1, b_l_out_1]

    return [bounds_0, bounds_1]


def backward_sort(
    inputs: List[Tensor],
    w_u_out: Tensor,
    b_u_out: Tensor,
    w_l_out: Tensor,
    b_l_out: Tensor,
    axis: int = -1,
    perturbation_domain: Optional[PerturbationDomain] = None,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    dc_decomp: bool = False,
) -> List[Tensor]:
    """Backward  LiRPA of sort

    Args:
        inputs
        w_u_out
        b_u_out
        w_l_out
        b_l_out
        axis
        perturbation_domain
        mode

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    mode = ForwardMode(mode)
    affine = get_affine(mode)
    z_value = K.cast(0.0, w_u_out.dtype)
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)

    # build fake inputs with no linearity
    n_dim = int(np.prod(u_c.shape[1:]))
    w_tmp = z_value * K.concatenate([u_c[:, None] * n_dim], 1)

    inputs_tmp = [x, u_c, w_tmp, u_c, l_c, w_tmp, l_c]
    outputs_tmp = sort(
        inputs_tmp, axis=axis, perturbation_domain=perturbation_domain, mode=ForwardMode.HYBRID, dc_decomp=False
    )
    _, _, w_u_tmp, b_u_tmp, _, w_l_tmp, b_l_tmp = outputs_tmp

    # w_u_tmp (None, n_dim, y.shape[1:)
    # w_u_out (None, 1, n_dim, n_out)
    w_u_tmp = K.reshape(w_u_tmp, [-1, n_dim, n_dim])  # (None, n_dim, n_dim)
    b_u_tmp = K.reshape(b_u_tmp, [-1, n_dim])  # (None, n_dim)
    w_l_tmp = K.reshape(w_l_tmp, [-1, n_dim, n_dim])
    b_l_tmp = K.reshape(b_l_tmp, [-1, n_dim])

    # combine with backward bounds
    w_u_out_pos = K.maximum(w_u_out, z_value)  # (None, 1, n_dim, n_out)
    w_u_out_neg = K.minimum(w_u_out, z_value)
    w_l_out_pos = K.maximum(w_l_out, z_value)
    w_l_out_neg = K.minimum(w_l_out, z_value)

    w_u_out = K.sum(w_u_out_pos * K.expand_dims(w_u_tmp, -1) + w_u_out_pos * K.expand_dims(w_l_tmp, -1), 1)
    w_l_out = K.sum(w_u_out_pos * K.expand_dims(w_u_tmp, -1) + w_u_out_pos * K.expand_dims(w_l_tmp, -1), 1)
    b_u_out = b_u_out + K.sum(w_u_out_pos * b_u_tmp, 1) + K.sum(w_u_out_neg * b_l_tmp, 1)
    b_l_out = b_l_out + K.sum(w_l_out_pos * b_l_tmp, 1) + K.sum(w_l_out_neg * b_u_tmp, 1)

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def get_identity_lirpa(inputs: List[Tensor]) -> List[Tensor]:
    y = inputs[-1]
    shape = int(np.prod(y.shape[1:]))

    y_flat = K.reshape(y, [-1, shape])

    w_u_out, w_l_out = [BatchedIdentityLike()(y_flat)] * 2
    b_u_out, b_l_out = [K.zeros_like(y_flat)] * 2

    return [w_u_out, b_u_out, w_l_out, b_l_out]


def get_identity_lirpa_shapes(input_shapes: List[Tuple[Optional[int], ...]]) -> List[Tuple[Optional[int], ...]]:
    y_shape = input_shapes[-1]
    batch_size = y_shape[0]
    flatten_dim = int(np.prod(y_shape[1:]))  # type: ignore

    b_shape = batch_size, flatten_dim
    w_shape = batch_size, flatten_dim, flatten_dim

    return [w_shape, b_shape, w_shape, b_shape]
