from decomon.types import Tensor

from .activation import DecomonBaseActivation
from .prime import leaky_relu_prime
from .utils import (
    get_convex_lower_affine_bound_unary,
    get_convex_upper_affine_bound_unary,
)


class DecomonReLU(DecomonBaseActivation):
    diagonal = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = self.layer.call

        raise NotImplementedError
        func_prime = lambda x: relu_prime(x)
