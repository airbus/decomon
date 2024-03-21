from typing import Any

import keras
import keras.ops as K

from decomon.keras_utils import add_tensors, batch_multid_dot
from decomon.layers.fuse import combine_affine_bounds
from decomon.layers.layer import DecomonLayer
from decomon.types import Tensor


class DecomonMerge(DecomonLayer):
    _is_merging_layer = True

    @property
    def keras_layer_input(self) -> list[keras.KerasTensor]:
        """self.layer.input returned as a list.

        In the degenerate case where only 1 input is merged, self.layer.input is a single keras tensor.
        We want here to have always a list, even with a single element, for consistency purpose.

        """
        if isinstance(self.layer.input, list):
            return self.layer.input
        else:
            return [self.layer.input]

    @property
    def nb_keras_inputs(self) -> int:
        """Number of inputs merged by the underlying layer."""
        return len(self.keras_layer_input)

    def get_affine_representation(self) -> tuple[list[Tensor], Tensor]:
        """Get affine representation of the layer

        This computes the affine representation of the layer, when this is meaningful,
        i.e. `self.linear` is True.

        If implemented, it will be used for backward and forward propagation of affine bounds through the layer.
        For non-linear layers, one should implement `get_affine_bounds()` instead.

        Args:

        Returns:
            [w_i]_i, b: affine representation of the layer satisfying

                layer(z) = Sum_i{w_i * z_i} + b

        More precisely, if the keras layer is merging inputs represented by z = (z_i)_i, we have
        ```
        layer(z) = sum(batch_multid_dot(z[i], w[i], missing_batchsize=(False, True)) for i in range(len(z)))  + b
        ```

        If w can be represented as a diagonal tensor, which means that the full version of w is retrieved by
            w_full = K.reshape(K.diag(K.flatten(w)), w.shape + w.shape)
        then
        - class attribute `diagonal` should be set to True (in order to have a correct `compute_output_shape()`)
        - we got
          ```
          layer(z) = sum(batch_multid_dot(z[i], w[i], missing_batchsize=(False, True), diagonal=(False, True)) for i in range(len(z)))+ b
          ```

        Shapes: !no batchsize!
            if diagonal is False:
                w[i]  ~ self.layer.input[i].shape[1:] + self.layer.output.shape[1:]
                b ~ self.layer.output.shape[1:]
            if diagonal is True:
                w[i]  ~ self.layer.output.shape[1:]
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

    def get_affine_bounds(
        self, lower: list[Tensor], upper: list[Tensor]
    ) -> tuple[list[Tensor], Tensor, list[Tensor], Tensor]:
        """Get affine bounds on layer outputs from layer inputs

        This compute the affine relaxation of the layer, given the oracle constant bounds on the inputs.

        If implemented, it will be used for backward and forward propagation of affine bounds through the layer.
        For linear layers, one can implement `get_affine_representation()` instead.

        Args:
            lower: lower constant oracle bounds on the layer inputs.
            upper: upper constant oracle bounds on the layer inputs.

        Returns:
            w_l, b_l, w_u, b_u: affine relaxation of the layer satisfying

                Sum_i{w_l[i] * z[i]} + b_l <= layer(z) <= Sum_i{w_u[i] * z[i]} + b_u
                with lower[i] <= z[i] <= upper[i]  for all i

        If w(_l or_u)[i] can be represented as a diagonal tensor, which means that the full version of w is retrieved by
            w_full = K.concatenate([K.reshape(K.diag(K.flatten(w[i])), w.shape + w.shape)[None] for i in range(len(w))], axis=0)
        then the class attribute `diagonal` should be set to True (in order to have a correct `compute_output_shape()`)

        Shapes:
            if diagonal is False:
                lowers[i], uppers[i] ~ (batchsize,) + self.layer.input[i].shape[1:]
                w_l[i], w_u[i]  ~ (batchsize,) + self.layer.input[i].shape[1:] + self.layer.output.shape[1:]
                b_l, b_u ~ (batchsize,) + self.layer.output.shape[1:]
            if diagonal is True:
                lower[i], upper[i] ~ (batchsize,) + self.layer.input[i].shape[1:]
                w_l[i], w_u[i]  ~ (batchsize,) + self.layer.output.shape[1:]
                b_l, b_u ~ (batchsize,) + self.layer.output.shape[1:]

        Note:
            `w * z` means here  `batch_multid_dot(z, w)`.

        """
        if self.linear:
            w, b = self.get_affine_representation()
            batchsize = lower[0].shape[0]
            w_with_batchsize = [K.repeat(w_i[None], batchsize, axis=0) for w_i in w]
            b_with_batchsize = K.repeat(b[None], batchsize, axis=0)
            return w_with_batchsize, b_with_batchsize, w_with_batchsize, b_with_batchsize
        else:
            raise NotImplementedError(
                "`get_affine_bounds()` needs to be implemented to get the forward and backward propagation of affine bounds. "
                "Alternatively, you can also directly override `forward_affine_propagate()` and `backward_affine_propagate()`"
            )

    def forward_ibp_propagate(self, lower: list[Tensor], upper: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Propagate ibp bounds through the layer.

        If the underlying keras layer is linear, it will be deduced from its affine representation.
        Else, this needs to be implemented to forward propagate ibp (constant) bounds.

        Args:
            lower: lower constant oracle bounds on the keras layer inputs.
            upper: upper constant oracle bounds on the keras layer inputs.

        Returns:
            l_c, u_c: constant relaxation of the layer satisfying
                l_c <= layer(z) <= u_c
                with lower[i] <= z[i] <= upper[i]  for all i

        Shapes:
            lower[i], upper[i] ~ (batchsize,) + self.layer.input[i].shape[1:]
            l_c, u_c ~ (batchsize,) + self.layer.output.shape[1:]

        """
        if self.linear:
            w, b = self.get_affine_representation()
            z_value = K.cast(0.0, dtype=b.dtype)

            l_c = b
            u_c = b

            for w_i, lower_i, upper_i in zip(w, lower, upper):
                is_diag = w_i.shape == b.shape
                kwargs_dot: dict[str, Any] = dict(missing_batchsize=(False, True), diagonal=(False, is_diag))

                w_i_pos = K.maximum(w_i, z_value)
                w_i_neg = K.minimum(w_i, z_value)

                # NB: += does not work well with broadcasting on pytorch backend => we use l_c = l_c + ...
                l_c = (
                    l_c
                    + batch_multid_dot(lower_i, w_i_pos, **kwargs_dot)
                    + batch_multid_dot(upper_i, w_i_neg, **kwargs_dot)
                )
                u_c = (
                    u_c
                    + batch_multid_dot(upper_i, w_i_pos, **kwargs_dot)
                    + batch_multid_dot(lower_i, w_i_neg, **kwargs_dot)
                )

            return l_c, u_c
        else:
            raise NotImplementedError(
                "`forward_ibp_propagate()` needs to be implemented to get the forward propagation of constant bounds."
            )

    def forward_affine_propagate(
        self, input_affine_bounds: list[list[Tensor]], input_constant_bounds: list[list[Tensor]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Propagate model affine bounds in forward direction.

        By default, this is deduced from `get_affine_bounds()` (or `get_affine_representation()` if `self.linear is True).
        But this could be overridden for better performance. See `DecomonConv2D` for an example.

        Args:
            input_affine_bounds[i]: [w_l_in[i], b_l_in[i], w_u_in[i], b_u_in[i]]
                affine bounds on underlying keras layer i-th input w.r.t. model input
            input_constant_bounds[i]: [l_c_in[i], u_c_in[i]]
                constant oracle bounds on underlying keras layer i-th input (already deduced from affine ones if necessary)

        Returns:
            w_l_new, b_l_new, w_u_new, b_u_new: affine bounds on underlying keras layer *output* w.r.t. model input

        If we denote by
          - x: keras model input
          - z[i]: underlying keras layer i-th input
          - h(x) = layer(z): output of the underlying keras layer

        The following inequations are satisfied
            w_l * x + b_l <= h(x) <= w_u * x + b_u
            l_c_in[i] <= z[i] <= u_c_in[i]  for all i
            w_l_in[i] * x + b_l_in[i] <= z[i] <= w_u_in[i] * x + b_u_in[i]  for all i

        """
        if self.linear:
            w, b = self.get_affine_representation()
            w_l, b_l, w_u, b_u = w, b, w, b
        else:
            lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)
            w_l, b_l, w_u, b_u = self.get_affine_bounds(lower=lower, upper=upper)

        b_l_new, b_u_new = b_l, b_u
        from_linear_layer_new = self.linear
        diagonal_w_new = True
        first_iteration = True
        for i in range(len(input_affine_bounds)):
            input_affine_bounds_i = input_affine_bounds[i]
            partial_layer_affine_bounds_i = [w_l[i], 0.0, w_u[i], 0.0]
            partial_layer_affine_bounds_true_shape_i = [w_l[i].shape, b_l.shape, w_u[i].shape, b_u.shape]
            from_linear_layer_combine = (
                self.inputs_outputs_spec.is_wo_batch_bounds(input_affine_bounds_i, i=i),
                self.linear,
            )
            diagonal_combine = (
                self.inputs_outputs_spec.is_diagonal_bounds(input_affine_bounds_i, i=i),
                self.inputs_outputs_spec.is_diagonal_bounds_shape(partial_layer_affine_bounds_true_shape_i, i=i),
            )
            delta_w_l_new, delta_b_l_new, delta_w_u_new, delta_b_u_new = combine_affine_bounds(
                affine_bounds_1=input_affine_bounds_i,
                affine_bounds_2=partial_layer_affine_bounds_i,
                from_linear_layer=from_linear_layer_combine,
                diagonal=diagonal_combine,
            )
            # add delta_b and delta_w, taking into account format (diagonal & from_linear_layer)
            from_linear_layer_delta = all(from_linear_layer_combine)
            diagonal_delta = all(diagonal_combine)
            from_linear_add = (from_linear_layer_new, from_linear_layer_delta)
            if first_iteration:
                w_l_new = delta_w_l_new
                w_u_new = delta_w_u_new
                first_iteration = False
                diagonal_w_new = diagonal_delta
            else:
                diagonal_add_w = (diagonal_w_new, diagonal_delta)
                w_l_new = add_tensors(
                    w_l_new,
                    delta_w_l_new,
                    missing_batchsize=from_linear_add,
                    diagonal=diagonal_add_w,
                )
                w_u_new = add_tensors(
                    w_u_new,
                    delta_w_u_new,
                    missing_batchsize=from_linear_add,
                    diagonal=diagonal_add_w,
                )
                diagonal_w_new = all(diagonal_add_w)
            b_l_new = add_tensors(
                b_l_new,
                delta_b_l_new,
                missing_batchsize=from_linear_add,
            )
            b_u_new = add_tensors(
                b_u_new,
                delta_b_u_new,
                missing_batchsize=from_linear_add,
            )
            from_linear_layer_new = all(from_linear_add)
        return w_l_new, b_l_new, w_u_new, b_u_new

    def backward_affine_propagate(  # type: ignore
        self, output_affine_bounds: list[Tensor], input_constant_bounds: list[list[Tensor]]
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Propagate model affine bounds in backward direction.

        By default, this is deduced from `get_affine_bounds()` (or `get_affine_representation()` if `self.linear is True).
        But this could be overridden for better performance. See `DecomonConv2D` for an example.

        Args:
            output_affine_bounds: [w_l, b_l, w_u, b_u]
                partial affine bounds on model output w.r.t underlying keras layer output
            input_constant_bounds: [[l_c_in[i], u_c_in[i]]]_i
                constant oracle bounds on underlying keras layer inputs

        Returns:
            [w_l_new[i], b_l_new[i], w_u_new[i], b_u_new[i]]_i: partial affine bounds on model output w.r.t. underlying keras layer *inputs*

        If we denote by
          - x: keras model input
          - m(x): keras model output
          - z[i]: underlying keras layer i-th input
          - h(x) = layer(z): output of the underlying keras layer
          - h_j(x) output of the j-th layer
          - w_l_j, b_l_j, w_u_j, b_u_j: current partial linear bounds on model output w.r.t to h_j(x)

        The following inequations are satisfied

            l_c_in[i] <= z[i] <= u_c_in[i]  for all i

            Sum_{others layers j}{w_l_j * h_j(x) + b_l_j} +  w_l * h(x) + b_l
              <= m(x)
              <= Sum_{others layers j}{w_u_j * h_j(x) + b_u_j} +  w_u * h(x) + b_u

            Sum_{others layers j}{w_l_j * h_j(x) + b_l_j} +  Sum_i{w_l_new[i] * z[i] + b_l_new[i]}
              <= m(x)
              <= Sum_{others layers j}{w_u_j * h_j(x) + b_u_j} +  Sum_i[w_u_new[i] * z[i] + b_u_new[i]}

        """
        if self.linear:
            w, b = self.get_affine_representation()
            w_l_layer, b_l_layer, w_u_layer, b_u_layer = w, b, w, b
        else:
            lower, upper = self.inputs_outputs_spec.split_constant_bounds(constant_bounds=input_constant_bounds)
            w_l_layer, b_l_layer, w_u_layer, b_u_layer = self.get_affine_bounds(lower=lower, upper=upper)

        from_linear_output_affine_bounds = self.inputs_outputs_spec.is_wo_batch_bounds(output_affine_bounds)
        diagonal_output_affine_bounds = self.inputs_outputs_spec.is_diagonal_bounds(output_affine_bounds)

        first_iteration = True
        propagated_affine_bounds = []
        output_affine_bounds_for_i = list(output_affine_bounds)
        for i in range(len(w_l_layer)):
            w_l_layer_i = w_l_layer[i]
            w_u_layer_i = w_u_layer[i]
            if first_iteration:
                # we mix the biases for first input
                b_l_layer_i = b_l_layer
                b_u_layer_i = b_u_layer
            else:
                # we skip the biases for the other inputs as already taken into account
                b_l_layer_i = K.zeros_like(b_l_layer)
                b_u_layer_i = K.zeros_like(b_u_layer)
                output_affine_bounds_for_i[1] = 0.0
                output_affine_bounds_for_i[3] = 0.0
            partial_layer_affine_bounds_i = [w_l_layer_i, b_l_layer_i, w_u_layer_i, b_u_layer_i]
            from_linear_layer = (self.linear, from_linear_output_affine_bounds)
            diagonal = (
                self.inputs_outputs_spec.is_diagonal_bounds(partial_layer_affine_bounds_i),
                diagonal_output_affine_bounds,
            )
            propagated_affine_bounds.append(
                combine_affine_bounds(
                    affine_bounds_1=partial_layer_affine_bounds_i,
                    affine_bounds_2=output_affine_bounds_for_i,
                    from_linear_layer=from_linear_layer,
                    diagonal=diagonal,
                )
            )
        return propagated_affine_bounds

    def get_forward_oracle(
        self,
        input_affine_bounds: list[list[Tensor]],
        input_constant_bounds: list[list[Tensor]],
        perturbation_domain_inputs: list[Tensor],
    ) -> list[list[Tensor]]:  # type: ignore
        """Get constant oracle bounds on underlying keras layer input from forward input bounds.

        Args:
            input_affine_bounds: affine bounds on each keras layer input w.r.t model input . Can be empty if not in affine mode.
            input_constant_bounds: ibp constant bounds on each keras layer input. Can be empty if not in ibp mode.
            perturbation_domain_inputs: perturbation domain input, wrapped in a list. Necessary only in affine mode, else empty.

        Returns:
            constant bounds on each keras layer input deduced from forward input bounds

        `input_affine_bounds, input_constant_bounds` are the forward bounds to be propagate through the layer.
        `input_affine_bounds` (resp. `input_constant_bounds`) will be empty if `self.affine` (resp. `self.ibp) is False.

        In hybrid case (ibp+affine), the constant bounds are assumed to be already tight, which means the previous
        forward layer should already have took the tighter constant bounds between the ibp ones and the ones deduced
        from the affine bounds given the considered perturbation domain.

        """
        return super().get_forward_oracle(input_affine_bounds=input_affine_bounds, input_constant_bounds=input_constant_bounds, perturbation_domain_inputs=perturbation_domain_inputs)  # type: ignore

    def call_forward(
        self,
        affine_bounds_to_propagate: list[list[Tensor]],
        input_bounds_to_propagate: list[list[Tensor]],
        perturbation_domain_inputs: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Propagate forward affine and constant bounds through the layer.

        Args:
            affine_bounds_to_propagate: affine bounds on each keras layer input w.r.t model input.
              Can be empty if not in affine mode.
            input_bounds_to_propagate: ibp constant bounds on each keras layer input. Can be empty if not in ibp mode.
            perturbation_domain_inputs: perturbation domain input, wrapped in a list. Necessary only in affine mode, else empty.

        Returns:
            output_affine_bounds, output_constant_bounds: affine and constant bounds on the underlying keras layer output

        """
        return super().call_forward(affine_bounds_to_propagate, input_bounds_to_propagate, perturbation_domain_inputs)

    def call_backward(
        self, affine_bounds_to_propagate: list[Tensor], constant_oracle_bounds: list[list[Tensor]]
    ) -> list[list[Tensor]]:
        return [
            list(partial_affine_bounds)
            for partial_affine_bounds in self.backward_affine_propagate(
                output_affine_bounds=affine_bounds_to_propagate, input_constant_bounds=constant_oracle_bounds
            )
        ]

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        """Propagate bounds in the specified direction `self.propagation`.

        Args:
            inputs: flattened decomon inputs
                - forward propagation: k affine bounds to propagate w.r.t. each keras layer input + k constant bounds

                    inputs = (
                        affine_bounds_to_propagate_0 +  constant_oracle_bounds_0 + ...
                        + affine_bounds_to_propagate_k +  constant_oracle_bounds_k
                        + perturbation_domain_inputs
                    )

                    with
                    - affine_bounds_to_propagate_* empty when affine is False;
                      (never to express identity bounds, as it would be impossible to separate bounds in flattened inputs)
                    - constant_oracle_bounds_* empty when ibp is False;
                    - perturbation_domain_inputs:
                        - if affine: the tensor defining the underlying keras model input perturbation, wrapped in a list;
                        - else: empty

                - backward propagation: only 1 affine bounds to propagate w.r.t keras layer output
                    + k constant bounds w.r.t each keras layer input

                    inputs = (
                        affine_bounds_to_propagate
                        + constant_oracle_bounds_0 + ... +  constant_oracle_bounds_k
                    )

                    with affine_bounds_to_propagate potentially empty to express identity affine bounds.

        Returns:
            the propagated bounds, in a flattened list.

            - in forward direction: affine_bounds_propagated + constant_bounds_propagated, each one being empty if self.affine or self.ibp is False
            - in backward direction: affine_bounds_propagated_0 + ... + affine_bounds_propagated_k, affine bounds w.r.t to each keras layer input

        """
        return super().call(inputs=inputs)
