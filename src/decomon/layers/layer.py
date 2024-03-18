from inspect import Parameter, signature
from typing import Any, Optional

import keras
import keras.ops as K
from keras.layers import Layer, Wrapper
from keras.utils import serialize_keras_object

from decomon.constants import Propagation
from decomon.layers.fuse import (
    combine_affine_bound_with_constant_bound,
    combine_affine_bounds,
)
from decomon.layers.inputs_outputs_specs import InputsOutputsSpec
from decomon.layers.oracle import get_forward_oracle
from decomon.perturbation_domain import BoxDomain, PerturbationDomain
from decomon.types import Tensor

_keras_base_layer_keyword_parameters = [
    name for name, param in signature(Layer.__init__).parameters.items() if param.kind == Parameter.KEYWORD_ONLY
] + ["input_shape", "input_dim"]


class DecomonLayer(Wrapper):
    """Base class for decomon layers.

    To enable LiRPA on a custom layer, one should implement a corresponding decomon layer by:
    - deriving from DecomonLayer
    - override/implement at least:
       - linear case:
         - set class attribute `linear` to True
         - `get_affine_representation()`
       - generic case:
         - `get_affine_bounds()`: affine bounds on layer output w.r.t. layer input
         - `forward_ibp_propagate()`: ibp bounds on layer ouput knowing ibp bounds on layer inpu
    - set class attribute `diagonal` to True if the affine relaxation is representing w as a diagonal
      (see explanation for `diagonal`).

    Other possibilities exist like overriding directly
        - `forward_affine_propagate()`
        - `backward_affine_propagate()`
        - `forward_ibp_propagate()` (still needed)

    """

    linear: bool = False
    """Flag telling that the layer is linear.

    Set it to True in child classes to explicit that the corresponding keras layer is linear.
    Else will be considered as non-linear.

    When linear is set to True, some computations can be simplified,
    even though equivalent with the ones made if linear is set to False.

    """

    diagonal: bool = False
    """Flag telling that the layer affine relaxation can be represented in a diagonal manner.

    This is useful only to compute properly output shapes without making actual computations, during call on symbolic tensors.
    During a call on actual tensors, this is not used.

    If diagonal is set to True, this means that if the affine relaxation of the layer is w_l, b_l, w_u, b_u then
    weights and bias will all have the same shape, and that the full representation can be retrieved

    - either in linear case with no batch axis by
      w_full = K.reshape(K.diag(K.flatten(w)))
    - or in generic case with a batch axis by the same computation, batch element by batch element, i.e.
      w_full = K.concatenate([K.reshape(K.diag(K.flatten(w[i])), w.shape + w.shape)[None] for i in range(len(w))], axis=0)

    In this case, the computations when merging affine bounds can be simplified.

    """

    _is_merging_layer: bool = False  # set to True in child class DecomonMerge

    def __init__(
        self,
        layer: Layer,
        perturbation_domain: Optional[PerturbationDomain] = None,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            layer: underlying keras layer
            perturbation_domain: default to a box domain
            ibp: if True, forward propagate constant bounds
            affine: if True, forward propagate affine bounds
            propagation: direction of bounds propagation
              - forward: from input to output
              - backward: from output to input
            model_output_shape: shape of the underlying model output (omitting batch axis).
               It allows determining if the backward bounds are with a batch axis or not.
            model_input_shape: shape of the underlying keras model input (omitting batch axis).
            **kwargs:

        """
        # Layer init:
        #  all options not expected by Layer are removed here
        #  (`to_decomon` pass options potentially not used by all decomon layers)
        keys_to_pop = [k for k in kwargs if k not in _keras_base_layer_keyword_parameters]
        for k in keys_to_pop:
            kwargs.pop(k)
        super().__init__(layer=layer, **kwargs)

        # default args
        if perturbation_domain is None:
            perturbation_domain = BoxDomain()

        # checks
        if not layer.built:
            raise ValueError(f"The underlying keras layer {layer.name} is not built.")
        if not ibp and not affine:
            raise ValueError("ibp and affine cannot be both False.")

        # attributes
        self.ibp = ibp
        self.affine = affine
        self.perturbation_domain = perturbation_domain
        self.propagation = propagation

        # input-output-manager
        self.inputs_outputs_spec = self.create_inputs_outputs_spec(
            layer=layer,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
        )

    def create_inputs_outputs_spec(
        self,
        layer: Layer,
        ibp: bool,
        affine: bool,
        propagation: Propagation,
        model_input_shape: Optional[tuple[int, ...]],
        model_output_shape: Optional[tuple[int, ...]],
    ) -> InputsOutputsSpec:
        if self._is_merging_layer:
            if isinstance(layer.input, keras.KerasTensor):
                # special case: merging a single input -> self.layer.input is already flattened
                layer_input_shape = [layer.input.shape[1:]]
            else:
                layer_input_shape = [t.shape[1:] for t in layer.input]
        else:
            layer_input_shape = layer.input.shape[1:]
        return InputsOutputsSpec(
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            layer_input_shape=layer_input_shape,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            is_merging_layer=self._is_merging_layer,
            linear=self.linear,
        )

    @property
    def is_merging_layer(self) -> bool:
        """Flag telling if the underlying keras layer is a merging layer or not (i.e. ~ with several inputs)."""
        return self._is_merging_layer

    @property
    def layer_input_shape(self) -> tuple[int, ...]:
        return self.inputs_outputs_spec.layer_input_shape

    @property
    def model_input_shape(self) -> tuple[int, ...]:
        return self.inputs_outputs_spec.model_input_shape

    @property
    def model_output_shape(self) -> tuple[int, ...]:
        return self.inputs_outputs_spec.model_output_shape

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "ibp": self.ibp,
                "affine": self.affine,
                "perturbation_domain": serialize_keras_object(self.perturbation_domain),
                "propagation": self.propagation,
                "model_input_shape": self.model_input_shape,
                "model_output_shape": self.model_output_shape,
            }
        )
        return config

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        """Get affine representation of the layer

        This computes the affine representation of the layer, when this is meaningful,
        i.e. `self.linear` is True.

        If implemented, it will be used for backward and forward propagation of affine bounds through the layer.
        For non-linear layers, one should implement `get_affine_bounds()` instead.

        Args:

        Returns:
            w, b: affine representation of the layer satisfying

                layer(z) = w * z + b

        More precisely, we have
        ```
        layer(z) = batch_multid_dot(z, w, missing_batchsize=(False, True)) + b
        ```

        If w can be represented as a diagonal tensor, which means that the full version of w is retrieved by
            w_full = K.reshape(K.diag(K.flatten(w)), w.shape + w.shape)
        then
        - class attribute `diagonal` should be set to True (in order to have a correct `compute_output_shape()`)
        - we got
          ```
          layer(z) = batch_multid_dot(z, w, missing_batchsize=(False, True), diagonal=(False, True)) + b
          ```

        Shapes: !no batchsize!
            if diagonal is False:
                w  ~ self.layer.input.shape[1:] + self.layer.output.shape[1:]
                b ~ self.layer.output.shape[1:]
            if diagonal is True:
                w  ~ self.layer.output.shape[1:]
                b ~ self.layer.output.shape[1:]


        """
        if not self.linear:
            raise RuntimeError("You should not call `get_affine_representation()` when `self.linear` is False.")
        else:
            raise NotImplementedError(
                "`get_affine_representation()` needs to be implemented to get the forward and backward propagation of affine bounds. "
                "Alternatively, you can also directly override "
                "`forward_ibp_propagate()`, `forward_affine_propagate()` and `backward_affine_propagate()`."
            )

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get affine bounds on layer outputs from layer inputs

        This compute the affine relaxation of the layer, given the oracle constant bounds on the inputs.

        If implemented, it will be used for backward and forward propagation of affine bounds through the layer.
        For linear layers, one can implement `get_affine_representation()` instead.

        Args:
            lower: lower constant oracle bound on the layer input.
            upper: upper constant oracle bound on the layer input.

        Returns:
            w_l, b_l, w_u, b_u: affine relaxation of the layer satisfying

                w_l * z + b_l <= layer(z) <= w_u * z + b_u
                with lower <= z <= upper

        If w(_l or_u) can be represented as a diagonal tensor, which means that the full version of w is retrieved by
            w_full = K.concatenate([K.reshape(K.diag(K.flatten(w[i])), w.shape + w.shape)[None] for i in range(len(w))], axis=0)
        then the class attribute `diagonal` should be set to True (in order to have a correct `compute_output_shape()`)

        Shapes:
            if diagonal is False:
                lower, upper ~ (batchsize,) + self.layer.input.shape[1:]
                w_l, w_u  ~ (batchsize,) + self.layer.input.shape[1:] + self.layer.output.shape[1:]
                b_l, b_u ~ (batchsize,) + self.layer.output.shape[1:]
            if diagonal is True:
                lower, upper ~ (batchsize,) + self.layer.input.shape[1:]
                w_l, w_u  ~ (batchsize,) + self.layer.output.shape[1:]
                b_l, b_u ~ (batchsize,) + self.layer.output.shape[1:]

        Note:
            `w * z` means here  `batch_multid_dot(z, w)`.

        """
        if self.linear:
            w, b = self.get_affine_representation()
            batchsize = lower.shape[0]
            w_with_batchsize = K.repeat(w[None], batchsize, axis=0)
            b_with_batchsize = K.repeat(b[None], batchsize, axis=0)
            return w_with_batchsize, b_with_batchsize, w_with_batchsize, b_with_batchsize
        else:
            raise NotImplementedError(
                "`get_affine_bounds()` needs to be implemented to get the forward and backward propagation of affine bounds. "
                "Alternatively, you can also directly override `forward_affine_propagate()` and `backward_affine_propagate()`"
            )

    def forward_ibp_propagate(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate ibp bounds through the layer.

        If the underlying keras layer is linear, it will be deduced from its affine representation.
        Else, this needs to be implemented to forward propagate ibp (constant) bounds.

        Args:
            lower: lower constant oracle bound on the keras layer input.
            upper: upper constant oracle bound on the keras layer input.

        Returns:
            l_c, u_c: constant relaxation of the layer satisfying
                l_c <= layer(z) <= u_c
                with lower <= z <= upper

        Shapes:
            lower, upper ~ (batchsize,) + self.layer.input.shape[1:]
            l_c, u_c ~ (batchsize,) + self.layer.output.shape[1:]

        """
        if self.linear:
            w, b = self.get_affine_representation()
            affine_bounds = [w, b, w, b]
            return combine_affine_bound_with_constant_bound(
                lower=lower, upper=upper, affine_bounds=affine_bounds, missing_batchsize=self.linear
            )
        else:
            raise NotImplementedError(
                "`forward_ibp_propagate()` needs to be implemented to get the forward propagation of constant bounds."
            )

    def forward_affine_propagate(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Propagate model affine bounds in forward direction.

        By default, this is deduced from `get_affine_bounds()` (or `get_affine_representation()` if `self.linear is True).
        But this could be overridden for better performance. See `DecomonConv2D` for an example.

        Args:
            input_affine_bounds: [w_l_in, b_l_in, w_u_in, b_u_in]
                affine bounds on underlying keras layer input w.r.t. model input
            input_constant_bounds: [l_c_in, u_c_in]
                constant oracle bounds on underlying keras layer input (already deduced from affine ones if necessary)

        Returns:
            w_l, b_l, w_u, b_u: affine bounds on underlying keras layer *output* w.r.t. model input

        If we denote by
          - x: keras model input
          - z: underlying keras layer input
          - h(x) = layer(z): output of the underlying keras layer

        The following inequations are satisfied
            w_l * x + b_l <= h(x) <= w_u * x + b_u
            l_c_in <= z <= u_c_in
            w_l_in * x + b_l_in <= z <= w_u_in * x + b_u_in

        """
        if self.linear:
            w, b = self.get_affine_representation()
            layer_affine_bounds = [w, b, w, b]
        else:
            lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower=lower, upper=upper)
            layer_affine_bounds = [w_l, b_l, w_u, b_u]

        from_linear_layer = (self.inputs_outputs_spec.is_wo_batch_bounds(input_affine_bounds), self.linear)
        diagonal = (
            self.inputs_outputs_spec.is_diagonal_bounds(input_affine_bounds),
            self.inputs_outputs_spec.is_diagonal_bounds(layer_affine_bounds),
        )
        return combine_affine_bounds(
            affine_bounds_1=input_affine_bounds,
            affine_bounds_2=layer_affine_bounds,
            from_linear_layer=from_linear_layer,
            diagonal=diagonal,
        )

    def backward_affine_propagate(
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Propagate model affine bounds in backward direction.

        By default, this is deduced from `get_affine_bounds()` (or `get_affine_representation()` if `self.linear is True).
        But this could be overridden for better performance. See `DecomonConv2D` for an example.

        Args:
            output_affine_bounds: [w_l, b_l, w_u, b_u]
                partial affine bounds on model output w.r.t underlying keras layer output
            input_constant_bounds: [l_c_in, u_c_in]
                constant oracle bounds on underlying keras layer input

        Returns:
            w_l_new, b_l_new, w_u_new, b_u_new: partial affine bounds on model output w.r.t. underlying keras layer *input*

        If we denote by
          - x: keras model input
          - m(x): keras model output
          - z: underlying keras layer input
          - h(x) = layer(z): output of the underlying keras layer
          - h_i(x) output of the i-th layer
          - w_l_i, b_l_i, w_u_i, b_u_i: current partial linear bounds on model output w.r.t to h_i(x)

        The following inequations are satisfied

            l_c_in <= z <= u_c_in

            Sum_{others layers i}(w_l_i * h_i(x) + b_l_i) +  w_l * h(x) + b_l
              <= m(x)
              <= Sum_{others layers i}(w_u_i * h_i(x) + b_u_i) +  w_u * h(x) + b_u

            Sum_{others layers i}(w_l_i * h_i(x) + b_l_i) +  w_l_new * z + b_l_new
              <= m(x)
              <= Sum_{others layers i}(w_u_i * h_i(x) + b_u_i) +  w_u_new * z + b_u_new

        """
        if self.linear:
            w, b = self.get_affine_representation()
            layer_affine_bounds = [w, b, w, b]
        else:
            lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower=lower, upper=upper)
            layer_affine_bounds = [w_l, b_l, w_u, b_u]

        from_linear_layer = (self.linear, self.inputs_outputs_spec.is_wo_batch_bounds((output_affine_bounds)))
        diagonal = (
            self.inputs_outputs_spec.is_diagonal_bounds(layer_affine_bounds),
            self.inputs_outputs_spec.is_diagonal_bounds(output_affine_bounds),
        )
        return combine_affine_bounds(
            affine_bounds_1=layer_affine_bounds,
            affine_bounds_2=output_affine_bounds,
            from_linear_layer=from_linear_layer,
            diagonal=diagonal,
        )

    def get_forward_oracle(
        self,
        input_affine_bounds: list[Tensor],
        input_constant_bounds: list[Tensor],
        perturbation_domain_inputs: list[Tensor],
    ) -> list[Tensor]:
        """Get constant oracle bounds on underlying keras layer input from forward input bounds.

        Args:
            input_affine_bounds: affine bounds on keras layer input w.r.t model input . Can be empty if not in affine mode.
            input_constant_bounds: ibp constant bounds on keras layer input. Can be empty if not in ibp mode.
            perturbation_domain_inputs: perturbation domain input, wrapped in a list. Necessary only in affine mode, else empty.

        Returns:
            constant bounds on keras layer input deduced from forward input bounds

        `input_affine_bounds, input_constant_bounds` are the forward bounds to be propagate through the layer.
        `input_affine_bounds` (resp. `input_constant_bounds`) will be empty if `self.affine` (resp. `self.ibp) is False.

        In hybrid case (ibp+affine), the constant bounds are assumed to be already tight, which means the previous
        forward layer should already have took the tighter constant bounds between the ibp ones and the ones deduced
        from the affine bounds given the considered perturbation domain.

        """
        from_linear = self.inputs_outputs_spec.is_wo_batch_bounds_by_keras_input(input_affine_bounds)
        return get_forward_oracle(
            affine_bounds=input_affine_bounds,
            ibp_bounds=input_constant_bounds,
            perturbation_domain_inputs=perturbation_domain_inputs,
            perturbation_domain=self.perturbation_domain,
            ibp=self.ibp,
            affine=self.affine,
            is_merging_layer=self.is_merging_layer,
            from_linear=from_linear,
        )

    def call_forward(
        self,
        affine_bounds_to_propagate: list[Tensor],
        input_bounds_to_propagate: list[Tensor],
        perturbation_domain_inputs: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Propagate forward affine and constant bounds through the layer.

        Args:
            affine_bounds_to_propagate: affine bounds on keras layer input w.r.t model input.
              Can be empty if not in affine mode.
              Can also be empty in case of identity affine bounds => we simply return layer affine bounds.
            input_bounds_to_propagate: ibp constant bounds on keras layer input. Can be empty if not in ibp mode.
            perturbation_domain_inputs: perturbation domain input, wrapped in a list. Necessary only in affine mode, else empty.

        Returns:
            output_affine_bounds, output_constant_bounds: affine and constant bounds on the underlying keras layer output

        Note:
            In hybrid case (ibp+affine), the constant bounds are assumed to be already tight in input, and we will return
            the tighter constant bounds in output. This means that
              - for the output: we take the tighter constant bounds between the ibp ones and the ones deduced
                  from the affine bounds given the considered perturbation domain, on the output.
              - for the input: we do not need it, as it should already have been taken care of in the previous layer

        """
        # IBP: interval bounds propragation
        if self.ibp:
            lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_bounds_to_propagate)
            output_constant_bounds = list(self.forward_ibp_propagate(lower=lower, upper=upper))
        else:
            output_constant_bounds = []

        # Affine bounds propagation
        if self.affine:
            if not self.linear:
                # get oracle input bounds (because input_bounds_to_propagate could be empty at this point)
                input_constant_bounds = self.get_forward_oracle(
                    input_affine_bounds=affine_bounds_to_propagate,
                    input_constant_bounds=input_bounds_to_propagate,
                    perturbation_domain_inputs=perturbation_domain_inputs,
                )
            else:
                input_constant_bounds = []
            # forward propagation
            output_affine_bounds = list(
                self.forward_affine_propagate(
                    input_affine_bounds=affine_bounds_to_propagate, input_constant_bounds=input_constant_bounds
                )
            )
        else:
            output_affine_bounds = []

        # Tighten constant bounds in hybrid mode (ibp+affine)
        if self.ibp and self.affine:
            if len(perturbation_domain_inputs) == 0:
                raise RuntimeError("keras model input is necessary for call_forward() in affine mode.")
            x = perturbation_domain_inputs[0]
            l_ibp, u_ibp = output_constant_bounds
            w_l, b_l, w_u, b_u = output_affine_bounds
            from_linear = self.linear and self.inputs_outputs_spec.is_wo_batch_bounds(
                affine_bounds=affine_bounds_to_propagate
            )
            l_affine = self.perturbation_domain.get_lower(x, w_l, b_l, missing_batchsize=from_linear)
            u_affine = self.perturbation_domain.get_upper(x, w_u, b_u, missing_batchsize=from_linear)
            u = K.minimum(u_ibp, u_affine)
            l = K.maximum(l_ibp, l_affine)
            output_constant_bounds = [l, u]

        return output_affine_bounds, output_constant_bounds

    def call_backward(
        self, affine_bounds_to_propagate: list[Tensor], constant_oracle_bounds: list[Tensor]
    ) -> list[Tensor]:
        return list(
            self.backward_affine_propagate(
                output_affine_bounds=affine_bounds_to_propagate, input_constant_bounds=constant_oracle_bounds
            )
        )

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        """Propagate bounds in the specified direction `self.propagation`.

        Args:
            inputs: concatenation of affine_bounds_to_propagate + constant_oracle_bounds + perturbation_domain_inputs with
                - affine_bounds_to_propagate: affine bounds to propagate.
                    Can be empty in forward direction if self.affine is False.
                    Can also be empty in case of identity affine bounds => we simply return layer affine bounds.
                - constant_oracle_bounds:
                    - in forward direction, the ibp bounds (empty if self.ibp is False);
                    - in backward direction, the oracle constant bounds on keras inputs (never empty)
                - perturbation_domain_inputs: the tensor defining the underlying keras model input perturbation, wrapped in a list.
                    - in forward direction when self.affine is True: a list with a single tensor x whose shape is given by `self.perturbation_domain.get_x_input_shape_wo_batchsize(model_input_shape)`
                      with `model_input_shape=model.input.shape[1:]` if `model` is the underlying keras model to analyse
                    - else: empty list

        Returns:
            the propagated bounds.
            - in forward direction: affine_bounds_propagated + constant_bounds_propagated, each one being empty if self.affine or self.ibp is False
            - in backward direction: affine_bounds_propagated

        """
        (
            affine_bounds_to_propagate,
            constant_oracle_bounds,
            perturbation_domain_inputs,
        ) = self.inputs_outputs_spec.split_inputs(inputs=inputs)
        if self.propagation == Propagation.FORWARD:  # forward
            affine_bounds_propagated, constant_bounds_propagated = self.call_forward(
                affine_bounds_to_propagate=affine_bounds_to_propagate,
                input_bounds_to_propagate=constant_oracle_bounds,
                perturbation_domain_inputs=perturbation_domain_inputs,
            )
            return self.inputs_outputs_spec.flatten_outputs(affine_bounds_propagated, constant_bounds_propagated)

        else:  # backward
            affine_bounds_propagated = self.call_backward(
                affine_bounds_to_propagate=affine_bounds_to_propagate, constant_oracle_bounds=constant_oracle_bounds
            )
            return self.inputs_outputs_spec.flatten_outputs(affine_bounds_propagated)

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        self.built = True

    def compute_output_shape(
        self,
        input_shape: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        (
            affine_bounds_to_propagate_shape,
            constant_oracle_bounds_shape,
            perturbation_domain_inputs_shape,
        ) = self.inputs_outputs_spec.split_input_shape(input_shape=input_shape)
        if self.propagation == Propagation.FORWARD:
            if self.ibp:
                constant_bounds_propagated_shape = [self.layer.output.shape] * 2
            else:
                constant_bounds_propagated_shape = []
            if self.affine:
                # layer output shape
                keras_layer_output_shape_wo_batchsize = self.layer.output.shape[1:]
                # model input shape
                model_input_shape_wo_batchsize = (
                    self.inputs_outputs_spec.model_input_shape
                )  # should be set to get accurate compute_output_shape()

                # outputs shape depends on layer and inputs being diagonal / linear (w/o batch)
                b_out_shape_wo_batchsize = keras_layer_output_shape_wo_batchsize
                if self.diagonal and self.inputs_outputs_spec.is_diagonal_bounds_shape(
                    affine_bounds_to_propagate_shape
                ):
                    # propagated bounds still diagonal
                    w_out_shape_wo_batchsize = b_out_shape_wo_batchsize
                else:
                    w_out_shape_wo_batchsize = model_input_shape_wo_batchsize + keras_layer_output_shape_wo_batchsize
                if self.linear and self.inputs_outputs_spec.is_wo_batch_bounds_shape(affine_bounds_to_propagate_shape):
                    # no batch in propagated bounds
                    w_out_shape = w_out_shape_wo_batchsize
                    b_out_shape = b_out_shape_wo_batchsize
                else:
                    w_out_shape = (None,) + w_out_shape_wo_batchsize
                    b_out_shape = (None,) + b_out_shape_wo_batchsize
                affine_bounds_propagated_shape = [w_out_shape, b_out_shape, w_out_shape, b_out_shape]
            else:
                affine_bounds_propagated_shape = []

            return self.inputs_outputs_spec.flatten_outputs_shape(
                affine_bounds_propagated_shape=affine_bounds_propagated_shape,
                constant_bounds_propagated_shape=constant_bounds_propagated_shape,
            )

        else:  # backward
            # model output shape
            model_output_shape_wo_batchsize = self.inputs_outputs_spec.model_output_shape
            # outputs shape depends if layer and inputs are diagonal / linear (w/o batch)
            b_shape_wo_batchisze = model_output_shape_wo_batchsize
            if self.diagonal and self.inputs_outputs_spec.is_diagonal_bounds_shape(affine_bounds_to_propagate_shape):
                if self._is_merging_layer:
                    w_shape_wo_batchsize = [model_output_shape_wo_batchsize] * self.inputs_outputs_spec.nb_keras_inputs
                else:
                    w_shape_wo_batchsize = model_output_shape_wo_batchsize
            else:
                if self._is_merging_layer:
                    w_shape_wo_batchsize = [
                        self.layer.input[i].shape[1:] + model_output_shape_wo_batchsize
                        for i in range(self.inputs_outputs_spec.nb_keras_inputs)
                    ]
                else:
                    w_shape_wo_batchsize = self.layer.input.shape[1:] + model_output_shape_wo_batchsize
            if self.linear and self.inputs_outputs_spec.is_wo_batch_bounds_shape(affine_bounds_to_propagate_shape):
                b_shape = b_shape_wo_batchisze
                w_shape = w_shape_wo_batchsize
            else:
                b_shape = (None,) + b_shape_wo_batchisze
                if self._is_merging_layer:
                    w_shape = [(None,) + sub_w_shape_wo_batchsize for sub_w_shape_wo_batchsize in w_shape_wo_batchsize]
                else:
                    w_shape = (None,) + w_shape_wo_batchsize
            if self._is_merging_layer:
                affine_bounds_propagated_shape = [
                    [
                        w_shape_i,
                        b_shape,
                        w_shape_i,
                        b_shape,
                    ]
                    for w_shape_i in w_shape
                ]
            else:
                affine_bounds_propagated_shape = [w_shape, b_shape, w_shape, b_shape]

            return self.inputs_outputs_spec.flatten_outputs_shape(
                affine_bounds_propagated_shape=affine_bounds_propagated_shape
            )
