from typing import Any

from decomon.types import Tensor

from .activation import DecomonBaseActivation


class DecomonReLU(DecomonBaseActivation):
    diagonal = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = self.layer.call

        raise NotImplementedError
