from collections.abc import Callable
from typing import Any, List, Optional

import keras
import keras.ops as K
import numpy as np
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
        self.alpha_custom = None
        if "alpha_custom" in kwargs:
            self.alpha_custom = kwargs["alpha_custom"]
        self.finetune = self.finetune and (self.finetune_lower or self.finetune_upper)

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

    def forward_affine_propagate(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_l_in, b_l_in, w_u_in, b_u_in = input_affine_bounds
        lower, upper = input_constant_bounds

        if self.finetune and self.diagonal:
            # warning not working if diagonal propagation !
            w_l_in_ = K.reshape(w_l_in, [-1] + list(self.model_input_shape) + self.layer_input_shape_wo_batchsize)
            w_u_in_ = K.reshape(w_u_in, [-1] + list(self.model_input_shape) + self.layer_input_shape_wo_batchsize)
            b_l_in_ = K.reshape(b_l_in, [-1] + self.layer_input_shape_wo_batchsize)
            b_u_in_ = K.reshape(b_u_in, [-1] + self.layer_input_shape_wo_batchsize)
            input_affine_bounds_ = [w_l_in_, b_l_in_, w_u_in_, b_u_in_]

            dico_alpha = {}
            if self.finetune_upper:
                dico_alpha["alpha_upper"] = self.alpha_upper(input_constant_bounds + input_affine_bounds_)
                dico_alpha["coeff_upper"] = self.coeff_upper
            if self.finetune_lower:
                dico_alpha["alpha_lower"] = self.alpha_lower(input_constant_bounds + input_affine_bounds_)
                dico_alpha["coeff_lower"] = self.coeff_lower

            # warning split into positive, negative if not fully increasing ... TODO

            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower, upper, finetune_forward=dico_alpha)
            w_l_ = K.reshape(w_l, [-1] + list(self.model_input_shape) + self.layer_input_shape_wo_batchsize)
            w_u_ = K.reshape(w_u, [-1] + list(self.model_input_shape) + self.layer_input_shape_wo_batchsize)

            if self.increasing:
                w_u_out = w_u_in * w_u_
                w_l_out = w_l_in * w_l_
                b_u_out = K.sum(w_u * K.expand_dims(b_u_in_, 1), 1) + b_u
                b_l_out = K.sum(w_l * K.expand_dims(b_l_in_, 1), 1) + b_l
            elif self.decreasing:
                w_u_out = w_l_in * w_u_
                w_l_out = w_u_in * w_l_
                b_u_out = K.sum(w_u * K.expand_dims(b_l_in_, 1), 1) + b_u
                b_l_out = K.sum(w_l * K.expand_dims(b_u_in_, 1), 1) + b_l
            else:
                w_u_pos = K.maximum(w_u_, 0)
                w_l_pos = K.maximum(w_l_, 0)
                w_u_neg = w_u - w_u_pos
                w_l_neg = w_l - w_l_pos

                w_u_out = w_u_in * w_u_pos + w_l_in * w_u_neg
                w_l_out = w_l_in * w_l_pos + w_u_in * w_l_neg
                b_u_out = (
                    K.sum(w_u_pos * K.expand_dims(b_u_in_, 1), 1) + K.sum(w_u_neg * K.expand_dims(b_l_in_, 1), 1) + b_u
                )
                b_l_out = (
                    K.sum(w_l_pos * K.expand_dims(b_l_in_, 1), 1) + K.sum(w_l_neg * K.expand_dims(b_u_in_, 1), 1) + b_l
                )

            return [w_l_out, b_l_out, w_u_out, b_u_out]
        else:
            return super().forward_affine_propagate(
                input_affine_bounds=input_affine_bounds, input_constant_bounds=input_constant_bounds
            )

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        

        if self.finetune and self.diagonal:
            w_l_out, b_l_out, w_u_out, b_u_out = output_affine_bounds
            lower, upper = input_constant_bounds
            is_from_linear = self.inputs_outputs_spec.is_wo_batch_bounds(output_affine_bounds)
            is_from_diagonal = self.inputs_outputs_spec.is_diagonal_bounds(output_affine_bounds)

            if is_from_linear:
                c_shape_wo_batch = list(b_l_out.shape)
            else:
                c_shape_wo_batch = list(b_l_out.shape[1:])

            if is_from_diagonal:
                w_l_out_ = K.reshape(w_l_out, [-1] + self.layer_output_shape_wo_batchsize + c_shape_wo_batch)
                w_u_out_ = K.reshape(w_u_out, [-1] + self.layer_output_shape_wo_batchsize + c_shape_wo_batch)
            else:
                w_l_out_ = K.reshape(w_l_out, [-1] + self.layer_output_shape_wo_batchsize + [1])
                w_u_out_ = K.reshape(w_u_out, [-1] + self.layer_output_shape_wo_batchsize + [1])

            output_affine_bounds_ = [w_l_out_, w_u_out_]

            dico_alpha = {}
            if self.finetune_upper:
                dico_alpha["alpha_upper"] = self.alpha_upper(input_constant_bounds + output_affine_bounds_)
                dico_alpha["coeff_upper"] = self.coeff_upper
            if self.finetune_lower:
                dico_alpha["alpha_lower"] = self.alpha_lower(input_constant_bounds + output_affine_bounds_)
                dico_alpha["coeff_lower"] = self.coeff_lower

            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower, upper, finetune_backward=dico_alpha)

            if is_from_diagonal:
                w_l_ = K.reshape(w_l, [-1] + self.layer_input_shape_wo_batchsize)
                w_u_ = K.reshape(w_u, [-1] + self.layer_input_shape_wo_batchsize)
                b_l_ = K.reshape(b_l, [-1] + self.layer_input_shape_wo_batchsize)
                b_u_ = K.reshape(b_u, [-1] + self.layer_input_shape_wo_batchsize)
            else:
                w_l_ = K.reshape(w_l, [-1] + self.layer_input_shape_wo_batchsize + c_shape_wo_batch)
                w_u_ = K.reshape(w_u, [-1] + self.layer_input_shape_wo_batchsize + c_shape_wo_batch)
                b_l_ = K.reshape(b_l, [-1] + self.layer_input_shape_wo_batchsize + c_shape_wo_batch)
                b_u_ = K.reshape(b_u, [-1] + self.layer_input_shape_wo_batchsize + c_shape_wo_batch)

            w_u_in = K.maximum(w_u_out, 0) * w_u_ + K.minimum(w_u_out, 0) * w_l_
            w_l_in = K.maximum(w_l_out, 0) * w_l_ + K.minimum(w_l_out, 0) * w_u_

            axis = [i + 1 for i in range(len(self.layer_output_shape_wo_batchsize))]

            b_u_in = K.sum(K.maximum(w_u_out, 0) * b_u_ + K.minimum(w_u_out, 0) * b_l_, axis) + b_u_out
            b_l_in = K.sum(K.maximum(w_l_out, 0) * b_l_ + K.minimum(w_l_out, 0) * b_u_, axis) + b_l_out

            return [w_l_in, b_l_in, w_u_in, b_u_in]
        else:
            return super().backward_affine_propagate(output_affine_bounds, input_constant_bounds)

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        super().build(input_shape=input_shape)
        if not self.linear and self.diagonal:
            input_shape_wo_batch = list(self.layer.input.shape[1:])
            model_input_shape = self.model_input_shape
            if self.finetune_upper and self.finetune:
                self.coeff_upper = self.add_weight(name="coeff_upper", shape=(1,), initializer="ones", trainable=True)
                self.alpha_upper = get_alpha_model_diagonal(
                    propagation=self.propagation,
                    input_shape_wo_batch=input_shape_wo_batch,
                    model_input_shape=model_input_shape,
                    alpha_model=self.alpha_custom,
                )
            if self.finetune_lower and self.finetune:
                self.coeff_lower = self.add_weight(name="coeff_lower", shape=(1,), initializer="ones", trainable=True)
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
            import pdb

            pdb.set_trace()
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

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = sigmoid
        func_prime = sigmoid_prime

        w_l, b_l, w_u, b_u = get_linear_hull_s_shape(lower, upper, func, func_prime)

        return w_l, b_l, w_u, b_u


class DecomonActivationTanh(DecomonBaseActivation):
    diagonal = True
    increasing = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = tanh
        func_prime = tanh_prime

        w_l, b_l, w_u, b_u = get_linear_hull_s_shape(lower, upper, func, func_prime)

        return w_l, b_l, w_u, b_u


class DecomonActivationExponential(DecomonBaseActivation):
    diagonal = True
    increasing = True
    # finetune_lower = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = exponential
        func_prime = exponential

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime, **kwargs)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope, **kwargs)

        return w_l, b_l, w_u, b_u


class DecomonActivationELU(DecomonBaseActivation):
    diagonal = True
    increasing = True
    # finetune_lower = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = elu
        func_prime = elu_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime, **kwargs)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope, **kwargs)

        return w_l, b_l, w_u, b_u


class DecomonActivationLeakyReLU(DecomonBaseActivation):
    diagonal = True
    increasing = True
    # finetune_lower = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = leaky_relu
        func_prime = leaky_relu_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime, **kwargs)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope, **kwargs)

        return w_l, b_l, w_u, b_u


class DecomonActivationSeLU(DecomonBaseActivation):
    diagonal = True
    increasing = True
    # finetune_lower = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = selu
        func_prime = selu_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime, **kwargs)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope, **kwargs)

        return w_l, b_l, w_u, b_u


class DecomonActivationSoftplus(DecomonBaseActivation):
    diagonal = True
    increasing = True
    finetune_lower = True

    def get_affine_bounds(self, lower: Tensor, upper: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        func = softplus
        func_prime = softplus_prime

        w_u, b_u = get_convex_upper_affine_bound_unary(lower, upper, func, func_prime, **kwargs)
        w_l, b_l = get_convex_lower_affine_bound_unary(lower, upper, func, func_prime, slope=self.slope, **kwargs)

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
