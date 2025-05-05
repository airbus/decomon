from typing import Any

from decomon.types import Tensor

from .activation import DecomonBaseActivation
from .prime import leaky_relu_prime
from .utils import (
    get_convex_lower_affine_bound_unary,
    get_convex_upper_affine_bound_unary,
)


class DecomonLeakyReLU(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = self.layer.call
        func_prime = lambda x: leaky_relu_prime(x, self.layer.negative_slope)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope)
        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime)

        return w_l, b_l, w_u, b_u
