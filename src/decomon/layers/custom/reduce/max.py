# define non native class Max
# Decomon Custom for Max(axis...)
from typing import Any

from keras_custom.layers import Max

from decomon.layers.layer import DecomonLayer
from decomon.types import Tensor

from .utils import (
    get_affine_lower_bound_max_before_reduction,
    get_affine_upper_bound_max_before_reduction,
    get_batch_multi_dot_repr_for_axis_reduce_weights,
)


class DecomonMax(DecomonLayer):
    layer: Max
    linear = False
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return get_affine_bounds_max(lower=lower, upper=upper, axis=self.layer.axis, keepdims=self.layer.keepdims)


def get_affine_bounds_max(
    lower: Tensor, upper: Tensor, axis: int, keepdims: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    w_l, b_l = get_affine_lower_bound_max_before_reduction(lower, upper, axis=axis, keepdims=keepdims)
    w_u, b_u = get_affine_upper_bound_max_before_reduction(lower, upper, axis=axis, keepdims=keepdims)

    # for now the lower bound is given by sum(w_l*x, axis=axis) + b,
    # so we need another transformation to get the final w_l
    w_l = get_batch_multi_dot_repr_for_axis_reduce_weights(w_l, axis=axis, keepdims=keepdims)
    w_u = get_batch_multi_dot_repr_for_axis_reduce_weights(w_u, axis=axis, keepdims=keepdims)

    return (w_l, b_l, w_u, b_u)
