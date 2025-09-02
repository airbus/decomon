from typing import Any, Optional

from keras.layers import Layer, SpectralNormalization

from decomon.constants import Propagation, Slope
from decomon.layers import DecomonLayer
from decomon.perturbation_domain import PerturbationDomain
from decomon.types import Tensor


class DecomonSpectralNormalization(DecomonLayer):
    layer: SpectralNormalization

    def __init__(
        self,
        layer: Layer,
        perturbation_domain: Optional[PerturbationDomain] = None,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
        slope: Slope = Slope.V_SLOPE,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            **kwargs,
        )

        """Performs spectral normalization on the weights of a target layer.

        This wrapper controls the Lipschitz constant of the weights of a layer by
        constraining their spectral norm, which can stabilize the training of GANs.
        Thus the affine relaxation will be the one of the sub_layer
        We need to assess whether the sub_layer is linear or not
        and convert it to its decomon counterpart
        Warning: we only support so fast sub layer that are native Keras layer which can be found automatically
        """

        self.sub_layer: Layer = self.layer.layer
        self.decomon_layer = self.layer

    def get_affine_representation(self, layer: Optional[Layer] = None) -> tuple[Tensor, Tensor]:
        return self.decomon_layer.get_affine_representation(layer=layer)

    def forward_ibp_propagate(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        return self.decomon_layer.forward_ibp_propagate(lower, upper)

    def forward_affine_propagate(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.decomon_layer.forward_affine_propagate(
            input_affine_bounds=input_affine_bounds, input_constant_bounds=input_constant_bounds
        )

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor], **kwargs: Any
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.decomon_layer.backward_affine_propagate(
            output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds, **kwargs
        )

    def get_forward_oracle(
        self,
        input_affine_bounds: list[Tensor],
        input_constant_bounds: list[Tensor],
        perturbation_domain_inputs: list[Tensor],
    ) -> list[Tensor]:
        return self.decomon_layer.get_forward_oracle(
            input_affine_bounds=input_affine_bounds,
            input_constant_bounds=input_constant_bounds,
            perturbation_domain_inputs=perturbation_domain_inputs,
        )

    def call_forward(
        self,
        affine_bounds_to_propagate: list[Tensor],
        input_bounds_to_propagate: list[Tensor],
        perturbation_domain_inputs: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        return self.call_forward(
            affine_bounds_to_propagate=affine_bounds_to_propagate,
            input_bounds_to_propagate=input_bounds_to_propagate,
            perturbation_domain_inputs=perturbation_domain_inputs,
        )

    def call_backward(
        self, affine_bounds_to_propagate: list[Tensor], constant_oracle_bounds: list[Tensor]
    ) -> list[Tensor]:
        return self.decomon_layer.call_backward(
            affine_bounds_to_propagate=affine_bounds_to_propagate, constant_oracle_bounds=constant_oracle_bounds
        )

    def call(self, inputs: list[Tensor], training: bool = False) -> list[Tensor]:
        return self.decomon_layer.call(inputs, training=training)
