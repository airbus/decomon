from typing import Any, Optional

from keras import Layer
from keras.layers import LeakyReLU

from decomon.constants import Propagation, Slope
from decomon.layers.layer import DecomonLayer
from decomon.perturbation_domain import PerturbationDomain
from decomon.types import Tensor

from .utils import get_leakyrelu_affine_bounds


class DecomonLeakyReLU(DecomonLayer):
    diagonal = True
    increasing = True
    layer: LeakyReLU

    def __init__(
        self,
        layer: Layer,
        perturbation_domain: Optional[PerturbationDomain] = None,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
        layer_backward: Optional[Layer] = None,
        layer_pos: Optional[Layer] = None,
        layer_neg: Optional[Layer] = None,
        finetune: bool = False,
        slope: Slope = Slope.V_SLOPE,
        **kwargs: Any,
    ):
        super().__init__(
            layer,
            perturbation_domain,
            ibp,
            affine,
            propagation,
            model_input_shape,
            model_output_shape,
            layer_backward,
            layer_pos,
            layer_neg,
            finetune,
            **kwargs,
        )
        self.slope = slope

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return get_leakyrelu_affine_bounds(
            lower=lower, upper=upper, slope=self.slope, negative_slope=self.layer.negative_slope, **kwargs
        )
