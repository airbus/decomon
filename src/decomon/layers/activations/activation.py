from collections.abc import Callable
from typing import Any, Optional

import keras
from keras import Layer
from keras.activations import linear, relu
from keras.layers import Activation

from decomon.core import PerturbationDomain, Propagation, Slope
from decomon.keras_utils import BatchedDiagLike
from decomon.layers.layer import DecomonLayer
from decomon.types import Tensor
from decomon.utils import get_linear_hull_relu


class DecomonBaseActivation(DecomonLayer):
    """Base class for decomon layers corresponding to activation layers."""

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
        self.slope = slope

    def forward_ibp_propagate(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate ibp bounds through the activation layer.

        By default, we simply apply the activation function on bounds.
        This is correct when the activation is an increasing function (like relu).
        This is not correct when the activation is not monotonic (like gelu).

        Args:
            lower:
            upper:

        Returns:

        """
        return self.layer.activation(lower), self.layer.activation(upper)


class DecomonActivation(DecomonBaseActivation):
    """Wrapping class for all decomon activation layer.

    Correspond to keras Activation layer.
    Will wrap a more specific activation Layer (DecomonRelu, DecomonLinear, ...)
    as it exists also a dedicated Relu layer in keras.

    """

    layer: Activation
    decomon_activation: DecomonBaseActivation

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
            slope=slope,
            **kwargs,
        )
        decomon_activation_class = get(self.layer.activation)
        self.decomon_activation = decomon_activation_class(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            slope=slope,
            **kwargs,
        )
        # linearity of the wrapping activation layer is decided by the wrapped activation layer
        self.linear = self.decomon_activation.linear
        # so do the inputs/outputs format
        self.inputs_outputs_spec = self.decomon_activation.inputs_outputs_spec

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        return self.decomon_activation.get_affine_representation()

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.decomon_activation.get_affine_bounds(lower=lower, upper=upper)

    def forward_affine_propagate(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.decomon_activation.forward_affine_propagate(
            input_affine_bounds=input_affine_bounds, input_constant_bounds=input_constant_bounds
        )

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.decomon_activation.backward_affine_propagate(
            output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds
        )

    def forward_ibp_propagate(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        return self.decomon_activation.forward_ibp_propagate(lower=lower, upper=upper)

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        self.decomon_activation.build(input_shape=input_shape)
        super().build(input_shape=input_shape)

    def compute_output_shape(self, input_shape: list[tuple[Optional[int], ...]]) -> list[tuple[Optional[int], ...]]:
        return self.decomon_activation.compute_output_shape(input_shape)

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        return self.decomon_activation.call(inputs=inputs)


class DecomonLinear(DecomonBaseActivation):
    linear = True

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        (
            affine_bounds_to_propagate,
            constant_oracle_bounds,
            perturbation_domain_inputs,
        ) = self.inputs_outputs_spec.split_inputs(inputs=inputs)
        return self.inputs_outputs_spec.flatten_outputs(
            affine_bounds_propagated=affine_bounds_to_propagate, constant_bounds_propagated=constant_oracle_bounds
        )

    def compute_output_spec(self, inputs: list[keras.KerasTensor]) -> list[keras.KerasTensor]:
        return self.call(inputs=inputs)

    def compute_output_shape(
        self,
        input_shape: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        (
            affine_bounds_to_propagate_shape,
            constant_oracle_bounds_shape,
            perturbation_domain_inputs_shape,
        ) = self.inputs_outputs_spec.split_input_shape(input_shape=input_shape)
        return self.inputs_outputs_spec.flatten_outputs_shape(
            affine_bounds_propagated_shape=affine_bounds_to_propagate_shape,
            constant_bounds_propagated_shape=constant_oracle_bounds_shape,
        )


class DecomonReLU(DecomonBaseActivation):
    diagonal = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_u, b_u, w_l, b_l = get_linear_hull_relu(upper=upper, lower=lower, slope=self.slope)
        return w_l, b_l, w_u, b_u


MAPPING_KERAS_ACTIVATION_TO_DECOMON_ACTIVATION: dict[Callable[[Tensor], Tensor], type[DecomonBaseActivation]] = {
    linear: DecomonLinear,
    relu: DecomonReLU,
}


def get(identifier: Any) -> type[DecomonBaseActivation]:
    """Retrieve a decomon activation layer via an identifier."""
    try:
        return MAPPING_KERAS_ACTIVATION_TO_DECOMON_ACTIVATION[identifier]
    except KeyError:
        raise NotImplementedError(f"No decomon layer existing for activation function {identifier}")
