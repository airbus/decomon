from inspect import Parameter, signature
from typing import Any, Optional, Union

import keras
import keras.ops as K
from keras.layers import Layer, Wrapper

from decomon.core import BoxDomain, PerturbationDomain, Propagation
from decomon.keras_utils import batch_multid_dot
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

    def __init__(
        self,
        layer: Layer,
        perturbation_domain: Optional[PerturbationDomain] = None,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        model_output_shape_length: int = 0,
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
            model_output_shape_length: length of the shape of the model output (omitting batch axis)
               w.r.t which backward affine bounds will be computed.
               It allows determining if the backward bounds are with a bacth axis or not.
               This has no meaning in forward propagation.
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
        if propagation == Propagation.BACKWARD and model_output_shape_length == 0:
            raise ValueError("model_output_shape_length must be positive in backward propagation.")

        # attributes
        self.ibp = ibp
        self.affine = affine
        self.perturbation_domain = perturbation_domain
        self.propagation = propagation
        self.model_output_shape_length = model_output_shape_length

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "ibp": self.ibp,
                "affine": self.affine,
                "perturbation_domain": self.perturbation_domain,
                "propagation": self.propagation,
                "model_output_shape_length": self.model_output_shape_length,
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
            is_diag = w.shape == b.shape
            kwargs_dot = dict(missing_batchsize=(False, True), diagonal=(False, is_diag))

            z_value = K.cast(0.0, dtype=w.dtype)
            w_pos = K.maximum(w, z_value)
            w_neg = K.minimum(w, z_value)

            l_c = batch_multid_dot(lower, w_pos, **kwargs_dot) + batch_multid_dot(upper, w_neg, **kwargs_dot) + b
            u_c = batch_multid_dot(upper, w_pos, **kwargs_dot) + batch_multid_dot(lower, w_neg, **kwargs_dot) + b

            return l_c, u_c
        else:
            raise NotImplementedError(
                "`forward_ibp_propagate()` needs to be implemented to get the forward propagation of constant bounds."
            )

    def forward_affine_propagate(
        self, input_affine_bounds, input_constant_bounds
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
            lower, upper = input_constant_bounds
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower=lower, upper=upper)
            layer_affine_bounds = [w_l, b_l, w_u, b_u]

        from_linear_layer = (self.is_wo_batch_bounds(input_affine_bounds), self.linear)

        return combine_affine_bounds(
            affine_bounds_1=input_affine_bounds,
            affine_bounds_2=layer_affine_bounds,
            from_linear_layer=from_linear_layer,
        )

    def backward_affine_propagate(
        self, output_affine_bounds, input_constant_bounds
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
            lower, upper = input_constant_bounds
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower=lower, upper=upper)
            layer_affine_bounds = [w_l, b_l, w_u, b_u]

        from_linear_layer = (self.linear, self.is_wo_batch_bounds((output_affine_bounds)))

        return combine_affine_bounds(
            affine_bounds_1=layer_affine_bounds,
            affine_bounds_2=output_affine_bounds,
            from_linear_layer=from_linear_layer,
        )

    def get_forward_oracle(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor], x: Tensor
    ) -> list[Tensor]:
        """Get constant oracle bounds on underlying keras layer input from forward input bounds.

        Args:
            input_affine_bounds: affine bounds on keras layer input w.r.t model input . Can be empty if not in affine mode.
            input_constant_bounds: ibp constant bounds on keras layer input. Can be empty if not in ibp mode.
            x: model input. Necessary only in affine mode.

        Returns:
            constant bounds on keras layer input deduced from forward input bounds

        `input_affine_bounds, input_constant_bounds` are the forward bounds to be propagate through the layer.
        `input_affine_bounds` (resp. `input_constant_bounds`) will be empty if `self.affine` (resp. `self.ibp) is False.

        In hybrid case (ibp+affine), the constant bounds are assumed to be already tight, which means the previous
        forward layer should already have took the tighter constant bounds between the ibp ones and the ones deduced
        from the affine bounds given the considered perturbation domain.

        """
        if self.ibp:
            # Hyp: in hybrid mode, the constant bounds are already tight
            # (affine and ibp mixed in forward layer output to get the tightest constant bounds)
            return input_constant_bounds

        elif self.affine:
            if len(input_affine_bounds) == 0:
                # special case: empty affine bounds => identity bounds
                l_affine = self.perturbation_domain.get_lower_x(x)
                u_affine = self.perturbation_domain.get_upper_x(x)
            else:
                w_l, b_l, w_u, b_u = input_affine_bounds
                l_affine = self.perturbation_domain.get_lower(x, w_l, b_l)
                u_affine = self.perturbation_domain.get_upper(x, w_u, b_u)
            return [l_affine, u_affine]

        else:
            raise RuntimeError("self.ibp and self.affine cannot be both False")

    def call_forward(
        self, affine_bounds_to_propagate: list[Tensor], input_bounds_to_propagate: list[Tensor], x: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Propagate forward affine and constant bounds through the layer.

        Args:
            affine_bounds_to_propagate: affine bounds on keras layer input w.r.t model input.
              Can be empty if not in affine mode.
              Can also be empty in case of identity affine bounds => we simply return layer affine bounds.
            input_bounds_to_propagate: ibp constant bounds on keras layer input. Can be empty if not in ibp mode.
            x: model input. Necessary only in affine mode.

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
            lower, upper = input_bounds_to_propagate
            output_constant_bounds = list(self.forward_ibp_propagate(lower=lower, upper=upper))
        else:
            output_constant_bounds = []

        # Affine bounds propagation
        if self.affine:
            if not self.linear:
                # get oracle input bounds (because input_bounds_to_propagate could be empty at this point)
                input_constant_bounds = self.get_forward_oracle(
                    input_affine_bounds=affine_bounds_to_propagate, input_constant_bounds=input_bounds_to_propagate, x=x
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
            l_ibp, u_ibp = output_constant_bounds
            w_l, b_l, w_u, b_u = output_affine_bounds
            l_affine = self.perturbation_domain.get_lower(x, w_l, b_l)
            u_affine = self.perturbation_domain.get_upper(x, w_u, b_u)
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

    def call(
        self, affine_bounds_to_propagate: list[Tensor], constant_oracle_bounds: list[Tensor], x: Tensor
    ) -> list[list[Tensor]]:
        """Propagate bounds in the specified direction `self.propagation`.

        Args:
            affine_bounds_to_propagate: affine bounds to propagate.
              Can be empty in forward direction if self.affine is False.
              Can also be empty in case of identity affine bounds => we simply return layer affine bounds.
            constant_oracle_bounds:  in forward direction, the ibp bounds (empty if self.ibp is False); in backward direction, the oracle constant bounds on keras inputs
            x: the model input. Necessary only in forward direction when self.affine is True.

        Returns:
            the propagated bounds.
            forward: [affine_bounds_propagated, constant_bounds_propagated], each one being empty if self.affine or self.ibp is False
            backward: [affine_bounds_propagated]


        """
        if self.propagation == Propagation.FORWARD:  # forward
            affine_bounds_propagated, constant_bounds_propagated = self.call_forward(
                affine_bounds_to_propagate=affine_bounds_to_propagate,
                input_bounds_to_propagate=constant_oracle_bounds,
                x=x,
            )
            return [affine_bounds_propagated, constant_bounds_propagated]

        else:  # backward
            affine_bounds_propagated = self.call_backward(
                affine_bounds_to_propagate=affine_bounds_to_propagate, constant_oracle_bounds=constant_oracle_bounds
            )
            return [affine_bounds_propagated]

    def build(self, affine_bounds_to_propagate_shape, constant_oracle_bounds_shape, x_shape):
        self.built = True

    def compute_output_shape(
        self,
        affine_bounds_to_propagate_shape: list[tuple[Optional[int], ...]],
        constant_oracle_bounds_shape: list[tuple[Optional[int], ...]],
        x_shape: tuple[Optional[int], ...],
    ):
        if self.propagation == Propagation.FORWARD:
            if self.ibp:
                constant_bounds_propagated_shape = [self.layer.output.shape] * 2
            else:
                constant_bounds_propagated_shape = []
            if self.affine:
                keras_layer_input_shape_wo_batchsize = self.layer.input.shape[1:]
                keras_layer_output_shape_wo_batchsize = self.layer.output.shape[1:]

                # inputs are in diagonal representation? without batch axis?
                if self.is_diagonal_bounds_shape(affine_bounds_to_propagate_shape):
                    model_input_shape_wo_batchsize = keras_layer_input_shape_wo_batchsize
                else:
                    w_in_shape = affine_bounds_to_propagate_shape[0]
                    if self.is_wo_batch_bounds_shape(affine_bounds_to_propagate_shape):
                        model_input_shape_wo_batchsize = w_in_shape[: -len(keras_layer_input_shape_wo_batchsize)]
                    else:
                        model_input_shape_wo_batchsize = w_in_shape[1 : -len(keras_layer_input_shape_wo_batchsize)]

                # outputs shape depends if layer and inputs are diagonal / linear (w/o batch)
                b_out_shape_wo_batchsize = keras_layer_output_shape_wo_batchsize
                if self.diagonal and self.is_diagonal_bounds_shape(affine_bounds_to_propagate_shape):
                    # propagated bounds still diagonal
                    w_out_shape_wo_batchsize = b_out_shape_wo_batchsize
                else:
                    w_out_shape_wo_batchsize = model_input_shape_wo_batchsize + keras_layer_output_shape_wo_batchsize
                if self.linear and self.is_wo_batch_bounds_shape(affine_bounds_to_propagate_shape):
                    # no batch in propagated bounds
                    w_out_shape = w_out_shape_wo_batchsize
                    b_out_shape = b_out_shape_wo_batchsize
                else:
                    w_out_shape = (None,) + w_out_shape_wo_batchsize
                    b_out_shape = (None,) + b_out_shape_wo_batchsize
                affine_bounds_propagated_shape = [w_out_shape, b_out_shape, w_out_shape, b_out_shape]
            else:
                affine_bounds_propagated_shape = []

            return [affine_bounds_propagated_shape, constant_bounds_propagated_shape]

        else:  # backward
            # find model output shape
            if self.is_identity_bounds_shape(affine_bounds_to_propagate_shape):
                model_output_shape_wo_batchsize = self.layer.output.shape[1:]
            else:
                b_shape = affine_bounds_to_propagate_shape[1]
                if self.is_wo_batch_bounds_shape(affine_bounds_to_propagate_shape):
                    model_output_shape_wo_batchsize = b_shape
                else:
                    model_output_shape_wo_batchsize = b_shape[1:]
            # outputs shape depends if layer and inputs are diagonal / linear (w/o batch)
            b_shape_wo_batchisze = model_output_shape_wo_batchsize
            if self.diagonal and self.is_diagonal_bounds_shape(affine_bounds_to_propagate_shape):
                w_shape_wo_batchsize = model_output_shape_wo_batchsize
            else:
                w_shape_wo_batchsize = self.layer.input.shape[1:] + model_output_shape_wo_batchsize
            if self.linear and self.is_wo_batch_bounds_shape(affine_bounds_to_propagate_shape):
                b_new_shape = b_shape_wo_batchisze
                w_shape = w_shape_wo_batchsize
            else:
                b_new_shape = (None,) + b_shape_wo_batchisze
                w_shape = (None,) + w_shape_wo_batchsize

            affine_bounds_propagated_shape = [w_shape, b_new_shape, w_shape, b_new_shape]

            return [affine_bounds_propagated_shape]

    def compute_output_spec(self, *args: Any, **kwargs: Any) -> list[list[keras.KerasTensor]]:
        """Compute output spec from output shape in case of symbolic call."""
        output_spec = Layer.compute_output_spec(self, *args, **kwargs)

        # fix empty list: Layer.compute_output_spec() transform them as empty tensors
        def replace_empty_tensor(l: Union[keras.KerasTensor, list[keras.KerasTensor]]):
            if isinstance(l, keras.KerasTensor) and len(l.shape) == 0:
                return []
            else:
                return l

        return [replace_empty_tensor(l) for l in output_spec]

    def is_identity_bounds(self, affine_bounds: list[Tensor]) -> bool:
        return len(affine_bounds) == 0

    def is_identity_bounds_shape(self, affine_bounds_shape: list[tuple[Optional[int], ...]]) -> bool:
        return len(affine_bounds_shape) == 0

    def is_diagonal_bounds(self, affine_bounds: list[Tensor]) -> bool:
        if self.is_identity_bounds(affine_bounds):
            return True
        w, b = affine_bounds[:2]
        return w.shape == b.shape

    def is_diagonal_bounds_shape(self, affine_bounds_shape: list[tuple[Optional[int], ...]]) -> bool:
        if self.is_identity_bounds_shape(affine_bounds_shape):
            return True
        w_shape, b_shape = affine_bounds_shape[:2]
        return w_shape == b_shape

    def is_wo_batch_bounds(self, affine_bounds: list[Tensor]) -> bool:
        if self.is_identity_bounds(affine_bounds):
            return True
        b = affine_bounds[1]
        if self.propagation == Propagation.FORWARD:
            return len(b.shape) == len(self.layer.input.shape) - 1
        else:
            return len(b.shape) == self.model_output_shape_length

    def is_wo_batch_bounds_shape(self, affine_bounds_shape: list[tuple[Optional[int], ...]]) -> bool:
        if self.is_identity_bounds_shape(affine_bounds_shape):
            return True
        b_shape = affine_bounds_shape[1]
        if self.propagation == Propagation.FORWARD:
            return len(b_shape) == len(self.layer.input.shape) - 1
        else:
            return len(b_shape) == self.model_output_shape_length


def combine_affine_bounds(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    from_linear_layer: tuple[bool, bool] = (False, False),
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_l_1, b_l_1, w_u_1, b_u_1] first affine bounds
        affine_bounds_2: [w_l_2, b_l_2, w_u_2, b_u_2] second affine bounds
        from_linear_layer: specify if affine_bounds_1 or affine_bounds_2
          come from the affine representation of a linear layer

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        w_l_1 * x + b_l_1 <= y <= w_u_1 * x + b_u_1
        w_l_2 * y + b_l_2 <= z <= w_u_2 * y + b_u_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u


    Special cases

    - with linear layers:

        If the affine bounds come from the affine representation of a linear layer (e.g. affine_bounds_1), then
          - lower and upper bounds are equal: affine_bounds_1 = [w_1, b_1, w_1, b_1]
          - the tensors are missing the batch dimension

        In the generic case, tensors in affine_bounds have their first axis corresponding to the batch size.

    - diagonal representation:

        If w.shape == b.shape, this means that w is represented by its "diagonal" (potentially a tensor-multid).

    - empty affine bounds:

        when one affine bounds is an empty list, this is actually a convention for identity bounds, i.e.
          w = identity, b = 0
        therefore we return the other affine_bounds, unchanged.

    """
    # special case: empty bounds <=> identity bounds
    if len(affine_bounds_1) == 0:
        return tuple(affine_bounds_2)
    if len(affine_bounds_2) == 0:
        return tuple(affine_bounds_1)

    # Are weights in diagonal representation?
    diagonal = (
        affine_bounds_1[0].shape == affine_bounds_1[1].shape,
        affine_bounds_2[0].shape == affine_bounds_2[1].shape,
    )
    if from_linear_layer == (False, False):
        return _combine_affine_bounds_generic(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )
    elif from_linear_layer == (True, False):
        return _combine_affine_bounds_left_from_linear(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )
    elif from_linear_layer == (False, True):
        return _combine_affine_bounds_right_from_linear(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )
    else:
        return _combine_affine_bounds_both_from_linear(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )


def _combine_affine_bounds_generic(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_l_1, b_l_1, w_u_1, b_u_1] first affine bounds
        affine_bounds_2: [w_l_2, b_l_2, w_u_2, b_u_2] second affine bounds
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        w_l_1 * x + b_l_1 <= y <= w_u_1 * x + b_u_1
        w_l_2 * y + b_l_2 <= z <= w_u_2 * x + b_u_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u

    """
    w_l_1, b_l_1, w_u_1, b_u_1 = affine_bounds_1
    w_l_2, b_l_2, w_u_2, b_u_2 = affine_bounds_2
    nb_axes_wo_batchsize_y = len(b_l_1.shape) - 1

    #  NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        diagonal=diagonal,
    )
    kwargs_dot_b = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        diagonal=(False, diagonal[1]),
    )

    z_value = K.cast(0.0, dtype=w_l_2.dtype)
    w_l_2_pos = K.maximum(w_l_2, z_value)
    w_u_2_pos = K.maximum(w_u_2, z_value)
    w_l_2_neg = K.minimum(w_l_2, z_value)
    w_u_2_neg = K.minimum(w_u_2, z_value)

    w_l = batch_multid_dot(w_l_1, w_l_2_pos, **kwargs_dot_w) + batch_multid_dot(w_u_1, w_l_2_neg, **kwargs_dot_w)
    w_u = batch_multid_dot(w_u_1, w_u_2_pos, **kwargs_dot_w) + batch_multid_dot(w_l_1, w_u_2_neg, **kwargs_dot_w)
    b_l = (
        batch_multid_dot(b_l_1, w_l_2_pos, **kwargs_dot_b) + batch_multid_dot(b_u_1, w_l_2_neg, **kwargs_dot_b) + b_l_2
    )
    b_u = (
        batch_multid_dot(b_u_1, w_u_2_pos, **kwargs_dot_b) + batch_multid_dot(b_l_1, w_u_2_neg, **kwargs_dot_b) + b_u_2
    )

    return w_l, b_l, w_u, b_u


def _combine_affine_bounds_right_from_linear(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_l_1, b_l_1, w_u_1, b_u_1] first affine bounds
        affine_bounds_2: [w_2, b_2, w_2, b_2] second affine bounds, with lower=upper + no batchsize
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        w_l_1 * x + b_l_1 <= y <= w_u_1 * x + b_u_1
        z = w_2 * y + b_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u

    """
    w_l_1, b_l_1, w_u_1, b_u_1 = affine_bounds_1
    w_2, b_2 = affine_bounds_2[:2]
    nb_axes_wo_batchsize_y = len(b_l_1.shape) - 1
    missing_batchsize = (False, True)

    #  NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )
    kwargs_dot_b = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=(False, diagonal[1]),
    )

    z_value = K.cast(0.0, dtype=w_2.dtype)
    w_2_pos = K.maximum(w_2, z_value)
    w_2_neg = K.minimum(w_2, z_value)

    w_l = batch_multid_dot(w_l_1, w_2_pos, **kwargs_dot_w) + batch_multid_dot(w_u_1, w_2_neg, **kwargs_dot_w)
    w_u = batch_multid_dot(w_u_1, w_2_pos, **kwargs_dot_w) + batch_multid_dot(w_l_1, w_2_neg, **kwargs_dot_w)
    b_l = batch_multid_dot(b_l_1, w_2_pos, **kwargs_dot_b) + batch_multid_dot(b_u_1, w_2_neg, **kwargs_dot_b) + b_2
    b_u = batch_multid_dot(b_u_1, w_2_pos, **kwargs_dot_b) + batch_multid_dot(b_l_1, w_2_neg, **kwargs_dot_b) + b_2

    return w_l, b_l, w_u, b_u


def _combine_affine_bounds_left_from_linear(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_1, b_1, w_1, b_1] first affine bounds, with lower=upper + no batchsize
        affine_bounds_2: [w_l_2, b_l_2, w_u_2, b_u_2] second affine bounds
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        y = w_1 * x + b_1
        w_l_2 * y + b_l_2 <= z <= w_u_2 * x + b_u_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u

    """
    w_1, b_1 = affine_bounds_1[:2]
    w_l_2, b_l_2, w_u_2, b_u_2 = affine_bounds_2
    nb_axes_wo_batchsize_y = len(b_1.shape)
    missing_batchsize = (True, False)

    #   NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )
    kwargs_dot_b = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=(False, diagonal[1]),
    )

    w_l = batch_multid_dot(w_1, w_l_2, **kwargs_dot_w)
    w_u = batch_multid_dot(w_1, w_u_2, **kwargs_dot_w)
    b_l = batch_multid_dot(b_1, w_l_2, **kwargs_dot_b) + b_l_2
    b_u = batch_multid_dot(b_1, w_u_2, **kwargs_dot_b) + b_u_2

    return w_l, b_l, w_u, b_u


def _combine_affine_bounds_both_from_linear(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_1, b_1, w_1, b_1] first affine bounds, with lower=upper + no batchsize
        affine_bounds_2: [w_2, b_2, w_2, b_2] second affine bounds, with lower=upper + no batchsize
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w, b, w, b: combined affine bounds

    If x, y, z satisfy
        y = w_1 * x + b_1
        z = w_2 * x + b_2

    Then
        z = w * x + b

    """
    w_1, b_1 = affine_bounds_1[:2]
    w_2, b_2 = affine_bounds_2[:2]
    nb_axes_wo_batchsize_y = len(b_1.shape)
    missing_batchsize = (True, True)

    #   NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )
    kwargs_dot_b = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=(False, diagonal[1]),
    )

    w = batch_multid_dot(w_1, w_2, **kwargs_dot_w)
    b = batch_multid_dot(b_1, w_2, **kwargs_dot_b) + b_2

    return w, b, w, b
