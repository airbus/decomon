from collections.abc import Callable
from typing import Any, List, Optional

import keras
import keras.ops as K
from keras import Layer
from keras.activations import (
    elu,
    exponential,
    leaky_relu,
    linear,
    relu,
    selu,
    sigmoid,
    softplus,
    softsign,
    tanh,
)
from keras.config import epsilon
from keras.layers import Activation

from decomon.constants import Propagation, Slope
from decomon.layers.activations.prime import (
    elu_prime,
    leaky_relu_prime,
    selu_prime,
    sigmoid_prime,
    softplus_prime,
    softsign_prime,
    tanh_prime,
)
from decomon.layers.activations.utils import (
    get_convex_lower_affine_bound_unary,
    get_convex_upper_affine_bound_unary,
    get_linear_hull_relu,
    get_linear_hull_s_shape,
)
from decomon.layers.finetune import get_alpha_model_diagonal
from decomon.layers.layer import DecomonLayer
from decomon.perturbation_domain import PerturbationDomain
from decomon.types import Tensor


class DecomonBaseActivation(DecomonLayer):
    """Base class for decomon layers corresponding to activation layers."""

    convex: bool = False
    concave: bool = False

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

        self.finetune = False
        self.alpha_custom = None
        if "alpha_custom" in kwargs:
            self.alpha_custom = kwargs["alpha_custom"]

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

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return super().backward_affine_propagate(output_affine_bounds, input_constant_bounds)

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        super().build(input_shape=input_shape)
        if not self.linear and self.diagonal:
            input_shape_wo_batch = list(self.layer.input.shape[1:])
            model_input_shape = self.model_input_shape
            if self.finetune_upper and self.finetune:
                self.alpha_upper = get_alpha_model_diagonal(
                    propagation=self.propagation,
                    input_shape_wo_batch=input_shape_wo_batch,
                    model_input_shape=model_input_shape,
                    alpha_model=self.alpha_custom,
                )
            if self.finetune_lower and self.finetune:
                self.alpha_lower = get_alpha_model_diagonal(
                    propagation=self.propagation,
                    input_shape_wo_batch=input_shape_wo_batch,
                    model_input_shape=model_input_shape,
                    alpha_model=self.alpha_custom,
                )


class DecomonActivation(DecomonBaseActivation):
    """Wrapping class for all decomon activation layer.

    Correspond to keras Activation layer.
    Will wrap a more specific activation Layer (DecomonRelu, DecomonLinear, ...)
    as it exists also a dedicated Relu layer in keras.

    """

    layer: Activation
    decomon_activation: DecomonBaseActivation
    diagonal = True
    increasing = True

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
        w_l_in, b_l_in, w_u_in, b_u_in = input_affine_bounds
        lower, upper = input_constant_bounds
        if self.finetune:
            alpha_models = [None, None]
            if self.finetune_upper:
                alpha_models[0] = self.alpha_upper
            if self.finetune_lower:
                alpha_models[1] = self.alpha_lower
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower, upper)
        if self.increasing:
            # weights are always positive
            if self.diagonal:
                import pdb

                pdb.set_trace()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        """
        return self.decomon_activation.forward_affine_propagate(
            input_affine_bounds=input_affine_bounds, input_constant_bounds=input_constant_bounds
        )
        """

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.decomon_activation.backward_affine_propagate(
            output_affine_bounds=output_affine_bounds, input_constant_bounds=input_constant_bounds
        )

    def forward_ibp_propagate(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        return self.decomon_activation.forward_ibp_propagate(lower=lower, upper=upper)

    def compute_output_shape(self, input_shape: list[tuple[Optional[int], ...]]) -> list[tuple[Optional[int], ...]]:
        return self.decomon_activation.compute_output_shape(input_shape)

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        return self.decomon_activation.call(inputs=inputs)

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        self.decomon_activation.build(input_shape=input_shape)


class DecomonLinear(DecomonBaseActivation):
    linear = True
    increasing = True

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
            constant_bounds_propagated_shape=constant_oracle_bounds_shape,  # type: ignore
        )


class DecomonActivationReLU(DecomonBaseActivation):
    diagonal = True
    increasing = True
    finetune_lower = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_u, b_u, w_l, b_l = get_linear_hull_relu(upper=upper, lower=lower, slope=self.slope, **kwargs)
        return w_l, b_l, w_u, b_u

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.finetune and not self.linear and self.diagonal:
            # do something
            if len(output_affine_bounds):
                is_from_linear = self.inputs_outputs_spec.is_wo_batch_bounds(output_affine_bounds)
                if is_from_linear:
                    output_affine_bounds = [K.expand_dims(e, 0) for e in output_affine_bounds]
                crown_inputs = output_affine_bounds[::2]

                batch_size = crown_inputs[0].shape[0]
                crown_inputs = [
                    K.reshape(e, [batch_size] + self.layer_output_shape_wo_batchsize + [-1])
                    + 0 * K.expand_dims(input_constant_bounds[0], -1)
                    for e in crown_inputs
                ]
                alpha_lower = self.alpha_lower(input_constant_bounds + crown_inputs)
                [w_l, b_l, w_u, b_u] = self.get_affine_bounds(
                    input_constant_bounds[0], input_constant_bounds[1], finetune={"alpha_lower": alpha_lower}
                )

                axis_b = [i + 1 for i in range(len(self.layer_input_shape_wo_batchsize))]
                backward_shape_wo_batch = list(output_affine_bounds[-1].shape[1:])

                w_u_out = K.reshape(
                    K.maximum(crown_inputs[1], 0) * w_u + K.minimum(crown_inputs[1], 0) * w_l,
                    [-1] + self.layer_input_shape_wo_batchsize + backward_shape_wo_batch,
                )
                b_u_out = (
                    K.reshape(
                        K.sum(K.maximum(crown_inputs[1], 0) * b_u + K.minimum(crown_inputs[1], 0) * b_l, axis=axis_b),
                        [-1] + backward_shape_wo_batch,
                    )
                    + output_affine_bounds[-1]
                )
                w_l_out = K.reshape(
                    K.maximum(crown_inputs[0], 0) * w_l + K.minimum(crown_inputs[0], 0) * w_u,
                    [-1] + self.layer_input_shape_wo_batchsize + backward_shape_wo_batch,
                )
                b_l_out = (
                    K.reshape(
                        K.sum(K.maximum(crown_inputs[0], 0) * b_u + K.minimum(crown_inputs[0], 0) * b_l, axis=axis_b),
                        [-1] + backward_shape_wo_batch,
                    )
                    + output_affine_bounds[1]
                )

                return [w_l_out, b_l_out, w_u_out, b_u_out]
                # layer
        else:
            return super().backward_affine_propagate(output_affine_bounds, input_constant_bounds)


class DecomonActivationSoftSign(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = softsign
        func_prime = softsign_prime

        # chord
        w_chord = (func(upper) - func(lower)) / K.maximum(K.cast(epsilon(), dtype=upper.dtype), upper - lower)
        b_chord = func(lower) - w_chord * lower

        # tangent at upper
        w_tangent_upper = func_prime(upper)
        b_tangent_upper = func(upper) - w_tangent_upper * upper

        # tangent at lower
        w_tangent_lower = func_prime(lower)
        b_tangent_lower = func(lower) - w_tangent_lower * lower

        # compare slopes to choose between chord and tangent
        w_l = K.where(
            w_chord <= w_tangent_lower,
            w_chord,
            w_tangent_lower,
        )
        b_l = K.where(
            w_chord <= w_tangent_lower,
            b_chord,
            b_tangent_lower,
        )

        w_u = K.where(
            w_chord <= w_tangent_upper,
            w_chord,
            w_tangent_upper,
        )
        b_u = K.where(
            w_chord <= w_tangent_upper,
            b_chord,
            b_tangent_upper,
        )

        return w_l, b_l, w_u, b_u


class DecomonActivationSigmoid(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = sigmoid
        func_prime = sigmoid_prime

        w_l, b_l, w_u, b_u = get_linear_hull_s_shape(lower, upper, func, func_prime)

        return w_l, b_l, w_u, b_u


class DecomonActivationTanh(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = tanh
        func_prime = tanh_prime

        w_l, b_l, w_u, b_u = get_linear_hull_s_shape(lower, upper, func, func_prime)

        return w_l, b_l, w_u, b_u


class DecomonActivationExponential(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = exponential
        func_prime = exponential

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope)

        return w_l, b_l, w_u, b_u


class DecomonActivationELU(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = elu
        func_prime = elu_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope)

        return w_l, b_l, w_u, b_u


class DecomonActivationLeakyReLU(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = leaky_relu
        func_prime = leaky_relu_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope)

        return w_l, b_l, w_u, b_u


class DecomonActivationSeLU(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = selu
        func_prime = selu_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope)

        return w_l, b_l, w_u, b_u


class DecomonActivationSoftplus(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = softplus
        func_prime = softplus_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope)

        return w_l, b_l, w_u, b_u


MAPPING_KERAS_ACTIVATION_TO_DECOMON_ACTIVATION: dict[Callable[[Tensor], Tensor], type[DecomonBaseActivation]] = {
    linear: DecomonLinear,
    relu: DecomonActivationReLU,
    softsign: DecomonActivationSoftSign,
    sigmoid: DecomonActivationSigmoid,
    tanh: DecomonActivationTanh,
    exponential: DecomonActivationExponential,
    elu: DecomonActivationELU,
    leaky_relu: DecomonActivationLeakyReLU,
    selu: DecomonActivationSeLU,
    softplus: DecomonActivationSoftplus,
}


def get(identifier: Any) -> type[DecomonBaseActivation]:
    """Retrieve a decomon activation layer via an identifier."""
    try:
        return MAPPING_KERAS_ACTIVATION_TO_DECOMON_ACTIVATION[identifier]
    except KeyError:
        raise NotImplementedError(f"No decomon layer existing for activation function {identifier}")
