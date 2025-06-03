# define non native class Max
# Decomon Custom for Max(axis...)
from typing import Any

import keras.ops as K
import numpy as np
from keras_custom.layers import Min

from decomon.layers.layer import DecomonLayer
from decomon.types import Tensor

from .max import get_affine_bounds_max
from .utils import (
    get_affine_lower_bound_max_before_reduction,
    get_affine_upper_bound_max_before_reduction,
)


class DecomonMin(DecomonLayer):
    r"""
    y = min(x)= -max(-x); h = max(-x), z= -x thus z \in [-upper, -lower]
    if w_l*z + b_l <= h <= w_u*z+b_u
    the following results hold: w_u*x - b_u <= y <= w_l*x-b_l
    """

    layer: Min
    linear = False
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_l_max, b_l_max, w_u_max, b_u_max = get_affine_bounds_max(
            lower=-upper, upper=-lower, axis=self.layer.axis, keepdims=self.layer.keepdims
        )
        w_l = w_u_max
        w_u = w_l_max
        b_l = -b_u_max
        b_u = -b_l_max

        return w_l, b_l, w_u, b_u
