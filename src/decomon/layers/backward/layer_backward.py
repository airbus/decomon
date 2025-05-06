# jacobinet module
from collections.abc import Callable
from typing import Any, Optional

import keras.ops as K
from jacobinet.layers import BackwardBoundedLinearizedLayer, BackwardLinearLayer
from jacobinet.layers.merging.base_merge import BackwardMergeLinearLayer
from keras.src.layers.merging.base_merge import Merge

from decomon.layers.layer import DecomonLinearLayer
from decomon.layers.merging.base_merge import DecomonMerge
from decomon.types import Tensor


class DecomonLinearLayerBackward(DecomonLinearLayer):
    "Use jacobinet to init layer_backward"

    linear: bool = True
    layer: BackwardLinearLayer

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        layer = kwargs["layer"]
        layer_backward = layer.layer
        super().__init__(*args, **kwargs)
        self.layer_backward = layer_backward

        if self.use_bias:
            self.bias = self.layer_backward(K.zeros([1] + self.layer.input_dim_wo_batch))

    def apply_layer_backward(self, input_: Tensor) -> Tensor:
        # remove bias before apply layer_backward
        if self.use_bias:
            return self.layer_backward(input_) - self.bias
        else:
            return self.layer_backward(input_)

    def compute_output_shape_forward(
        self,
        input_shape: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        (
            affine_bounds_to_propagate_shape,
            _,
            _,
        ) = self.inputs_outputs_spec.split_input_shape(input_shape=input_shape)
        keras_layer_output_shape_wo_batchsize = self.layer.input_dim_wo_batch
        affine_bounds_propagated_shape: list[tuple[Optional[int], ...]]
        constant_bounds_propagated_shape: list[tuple[Optional[int], ...]]
        if self.ibp:
            if isinstance(self.layer_backward, Merge):
                # BackwardLayer returns a list of list
                constant_bounds_propagated_shape = [
                    [[1] + output_dim_i] * 2 for output_dim_i in keras_layer_output_shape_wo_batchsize
                ]
            else:
                constant_bounds_propagated_shape = [self.layer.output.shape] * 2
        else:
            constant_bounds_propagated_shape = []
        if self.affine:
            # temporary patch
            if isinstance(affine_bounds_to_propagate_shape[0], list):
                affine_bounds_to_propagate_shape = affine_bounds_to_propagate_shape[0]

            # layer output shape
            keras_layer_output_shape_wo_batchsize = self.layer.output.shape[1:]
            # model input shape
            model_input_shape_wo_batchsize = (
                self.inputs_outputs_spec.model_input_shape
            )  # should be set to get accurate compute_output_shape()

            # outputs shape depends on layer and inputs being diagonal / linear (w/o batch)
            b_out_shape_wo_batchsize = keras_layer_output_shape_wo_batchsize

            if self.diagonal and self.inputs_outputs_spec.is_diagonal_bounds_shape(affine_bounds_to_propagate_shape):
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


class DecomonBoundedLinearizedLayerBackward(DecomonLinearLayer):
    "Use jacobinet to init layer_backward"

    linear: bool = True
    layer: BackwardBoundedLinearizedLayer  # add in jacobinet
    increasing = True  # temporary

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.layer_up = self.layer.layer_backward_up
        self.layer_low = self.layer.layer_backward_low
        self.layer_backward_up = self.layer.layer_up
        self.layer_backward_low = self.layer.layer_low

        if self.use_bias:
            self.bias_up = self.layer_up(K.zeros([1] + self.layer.output_dim_wo_batch))
            self.bias_low = self.layer_low(K.zeros([1] + self.layer.output_dim_wo_batch))

    def apply_layer_backward_upper(self, input_: Tensor) -> Tensor:
        # remove bias before apply layer_backward
        if self.increasing:
            if self.use_bias:
                return self.layer_up(input_) - self.bias
            else:
                return self.layer_up(input_)
        else:
            if self.use_bias:
                raise NotImplementedError()
            else:
                return self.layer_up(K.relu(input_)) + self.layer_low(-K.relu(-input_))

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
        if self.increasing:
            lower_bound = self.layer_low(lower)
            upper_bound = self.layer_up(upper)

            return (lower_bound, upper_bound)

        else:
            raise NotImplementedError()

    def forward_affine_propagate(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_l_in, b_l_in, w_u_in, b_u_in = input_affine_bounds
        # reshape
        w_l_in_ = K.reshape(w_l_in, [-1] + self.layer_input_shape_wo_batchsize)
        w_u_in_ = K.reshape(w_u_in, [-1] + self.layer_input_shape_wo_batchsize)
        b_l_in_ = K.reshape(b_l_in, [-1] + self.layer_input_shape_wo_batchsize)
        b_u_in_ = K.reshape(b_u_in, [-1] + self.layer_input_shape_wo_batchsize)

        if self.increasing:
            output_shape = [-1] + list(self.model_input_shape) + self.layer_output_shape_wo_batchsize
            w_l_out = K.reshape(self.layer_low(w_l_in_), output_shape)
            b_l_out = self.layer_low(b_l_in_)
            w_u_out = K.reshape(self.layer_up(w_u_in_), output_shape)
            b_u_out = self.layer_up(b_u_in_)

            return (w_l_out, b_l_out, w_u_out, b_u_out)
        else:
            raise NotImplementedError()


class DecomonLinearMergeBackward(DecomonLinearLayer):
    "a backxard merge layer take one tensor as input and return a list of tensor"

    linear: bool = True
    layer: BackwardMergeLinearLayer

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        layer = kwargs["layer"]
        layer_backward = layer.layer
        super().__init__(*args, **kwargs)
        self.layer_backward = layer_backward

        self.bias = self.layer_backward(
            [K.zeros([1] + input_dim_wo_batch_i) for input_dim_wo_batch_i in self.layer.input_dim_wo_batch]
        )

    def apply_layer_backward(self, input_: Tensor) -> Tensor:
        # remove bias before apply layer_backward
        raise NotImplementedError()

    @property
    def layer_output_shape_wo_batchsize(self) -> list[list[int]]:
        return [list(e.shape[1:]) for e in self.layer.output]

    def compute_output_shape_forward(
        self,
        input_shape: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        (
            affine_bounds_to_propagate_shape,
            _,
            _,
        ) = self.inputs_outputs_spec.split_input_shape(input_shape=input_shape)
        keras_layer_output_shape_wo_batchsize = self.layer.input_dim_wo_batch
        constant_bounds_propagated_shape: list[tuple[Optional[int], ...]]
        if self.ibp:
            constant_bounds_propagated_shape = [
                [(1,) + output_dim_i] * 2 for output_dim_i in keras_layer_output_shape_wo_batchsize
            ]
        else:
            constant_bounds_propagated_shape = []
        if self.affine:
            # model input shape
            model_input_shape_wo_batchsize = list(self.inputs_outputs_spec.model_input_shape)
            # should be set to get accurate compute_output_shape()

            # outputs shape depends on layer and inputs being diagonal / linear (w/o batch)
            b_out_shape_wo_batchsize = keras_layer_output_shape_wo_batchsize  # list

            if self.diagonal and self.inputs_outputs_spec.is_diagonal_bounds_shape(affine_bounds_to_propagate_shape):
                # propagated bounds still diagonal
                w_out_shape_wo_batchsize = b_out_shape_wo_batchsize
            else:
                w_out_shape_wo_batchsize = [
                    model_input_shape_wo_batchsize + w_out_i for w_out_i in keras_layer_output_shape_wo_batchsize
                ]

            if self.linear and self.inputs_outputs_spec.is_wo_batch_bounds_shape(affine_bounds_to_propagate_shape):
                # no batch in propagated bounds
                w_out_shape = w_out_shape_wo_batchsize
                b_out_shape = b_out_shape_wo_batchsize
            else:
                w_out_shape = [
                    [None] + list(w_out_i) for w_out_i in w_out_shape_wo_batchsize
                ]  # (None,) + w_out_shape_wo_batchsize
                b_out_shape = [
                    [None] + list(b_out_i) for b_out_i in b_out_shape_wo_batchsize
                ]  # (None,) + b_out_shape_wo_batchsize

            affine_bounds_propagated_shape = [
                [w_out_shape_i, b_out_shape_i, w_out_shape_i, b_out_shape_i]
                for (w_out_shape_i, b_out_shape_i) in zip(w_out_shape, b_out_shape)
            ]
        else:
            affine_bounds_propagated_shape = []

        return self.inputs_outputs_spec.flatten_outputs_shape(
            affine_bounds_propagated_shape=affine_bounds_propagated_shape,
            constant_bounds_propagated_shape=constant_bounds_propagated_shape,
        )

    def forward_affine_propagate(
        self, input_affine_bounds: list[Tensor], input_constant_bounds: list[Tensor]
    ) -> list[list[Tensor]]:  # type: ignore
        w_l_in, b_l_in, w_u_in, b_u_in = input_affine_bounds
        is_from_linear = self.inputs_outputs_spec.is_wo_batch_bounds(input_affine_bounds)

        # reshape
        w_l_in_ = K.reshape(w_l_in, [-1] + self.layer_input_shape_wo_batchsize)
        w_u_in_ = K.reshape(w_u_in, [-1] + self.layer_input_shape_wo_batchsize)
        b_l_in_ = K.reshape(b_l_in, [-1] + self.layer_input_shape_wo_batchsize)
        b_u_in_ = K.reshape(b_u_in, [-1] + self.layer_input_shape_wo_batchsize)

        def apply_layer_on_w(
            w_in: Tensor, func: Callable[[Tensor], list[Tensor]], output_shape: list[list[int]]
        ) -> list[Tensor]:
            w_out_list = func(w_in)
            return [K.reshape(w_out_i, output_shape_i) for (w_out_i, output_shape_i) in zip(w_out_list, output_shape)]

        if is_from_linear:
            # apply layer
            w_l_out = apply_layer_on_w(
                w_u_in_, self.layer, [list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize]
            )
            w_u_out = apply_layer_on_w(
                w_l_in_, self.layer, [list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize]
            )
            b_l_out = [b_i[0] for b_i in self.layer(b_l_in_)]
            b_u_out = [b_i[0] for b_i in self.layer(b_u_in_)]

            return [
                [w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i]
                for (w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i) in zip(w_l_out, b_l_out, w_u_out, b_u_out)
            ]
        else:
            if self.increasing:
                w_l_out = apply_layer_on_w(
                    w_l_in_,
                    self.layer,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )
                w_u_out = apply_layer_on_w(
                    w_u_in_,
                    self.layer,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )
                b_l_out = self.layer(b_l_in)
                b_u_out = self.layer(b_u_in)

                return [
                    [w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i]
                    for (w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i) in zip(w_l_out, b_l_out, w_u_out, b_u_out)
                ]

            if self.decreasing:
                w_l_out = apply_layer_on_w(
                    w_u_in_,
                    self.layer,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )
                w_u_out = apply_layer_on_w(
                    w_l_in_,
                    self.layer,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )
                b_l_out = self.layer(b_l_in)
                b_u_out = self.layer(b_u_in)

                return [
                    [w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i]
                    for (w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i) in zip(w_l_out, b_l_out, w_u_out, b_u_out)
                ]

            if not (self.layer_pos is None) and not (self.layer_neg is None):
                w_l_out_pos = apply_layer_on_w(
                    w_l_in_,
                    self.layer_pos,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )
                w_u_out_pos = apply_layer_on_w(
                    w_u_in_,
                    self.layer_pos,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )
                w_l_out_neg = apply_layer_on_w(
                    w_u_in_,
                    self.layer_neg,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )
                w_u_out_neg = apply_layer_on_w(
                    w_l_in_,
                    self.layer_neg,
                    [[-1] + list(self.model_input_shape) + e for e in self.layer_output_shape_wo_batchsize],
                )

                w_u_out = [w_u_p + w_u_n for (w_u_p, w_u_n) in zip(w_u_out_pos, w_u_out_neg)]
                w_l_out = [w_l_p + w_l_n for (w_l_p, w_l_n) in zip(w_l_out_pos, w_l_out_neg)]

                b_l_out_pos = self.layer_pos(b_l_in)
                b_l_out_neg = self.layer_neg(b_u_in)
                b_l_out = [b_l_p + b_l_n for (b_l_p, b_l_n) in zip(b_l_out_pos, b_l_out_neg)]

                b_u_out_pos = self.layer_pos(b_u_in)
                b_u_out_neg = self.layer_neg(b_l_in)
                b_u_out = [b_u_p + b_u_n for (b_u_p, b_u_n) in zip(b_u_out_pos, b_u_out_neg)]

                return [
                    [w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i]
                    for (w_l_out_i, b_l_out_i, w_u_out_i, b_u_out_i) in zip(w_l_out, b_l_out, w_u_out, b_u_out)
                ]

            else:
                return super().forward_affine_propagate(input_affine_bounds, input_constant_bounds)

    def forward_ibp_propagate(self, lower: list[Tensor], upper: list[Tensor]) -> list[list[Tensor]]:  # type: ignore
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
        if self.increasing:
            lower_bounds = self.layer(lower)
            upper_bounds = self.layer(upper)
            return [[l, u] for (l, u) in zip(lower_bounds, upper_bounds)]

        if self.decreasing:
            lower_bounds = self.layer(lower)
            upper_bounds = self.layer(upper)
            return [[u, l] for (l, u) in zip(lower_bounds, upper_bounds)]

        else:
            raise NotImplementedError(
                "`forward_ibp_propagate()` needs to be implemented to get the forward propagation of constant bounds."
            )


class DecomonNonLinearBackward(DecomonMerge):
    linear = False
    increasing = True

    def forward_ibp_propagate(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        # temporary
        g_lower, g_upper = lower[0], upper[0]
        input_lower, input_upper = lower[1], upper[1]

        upper = self.layer([g_upper, input_upper])
        lower = self.layer([g_lower, input_lower])
        return (lower, upper)
