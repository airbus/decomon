from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dot,
    Dropout,
    Flatten,
    InputLayer,
    InputSpec,
    Lambda,
    Layer,
)
from tensorflow.python.keras.utils import conv_utils

from decomon.core import ForwardMode, PerturbationDomain, Slope
from decomon.layers import activations
from decomon.layers.core import DecomonLayer
from decomon.layers.utils import (
    ClipAlpha,
    ClipAlphaAndSumtoOne,
    NonPos,
    Project_initializer_pos,
)


class DecomonConv2D(DecomonLayer, Conv2D):
    """Forward LiRPA implementation of Conv2d layers.
    See Keras official documentation for further details on the Conv2d operator

    """

    original_keras_layer_class = Conv2D

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        kwargs.pop("activation", None)
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        if self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=2),  # z
                InputSpec(min_ndim=4),  # u
                InputSpec(min_ndim=4),  # wu
                InputSpec(min_ndim=4),  # bu
                InputSpec(min_ndim=4),  # l
                InputSpec(min_ndim=4),  # wl
                InputSpec(min_ndim=4),  # bl
            ]
        elif self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(min_ndim=4),  # u
                InputSpec(min_ndim=4),  # l
            ]
        elif self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=2),  # z
                InputSpec(min_ndim=4),  # wu
                InputSpec(min_ndim=4),  # bu
                InputSpec(min_ndim=4),  # wl
                InputSpec(min_ndim=4),  # bl
            ]
        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=4), InputSpec(min_ndim=4)]

        self.diag_op = Lambda(tf.linalg.diag)

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape

        Returns:

        """

        assert len(input_shape) == self.nb_tensors
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError("The channel dimension of the inputs " "should be defined. Found `None`.")
        input_dim = input_shape[-1][channel_axis]

        kernel_shape = self.kernel_size + (input_dim, self.filters)

        if not self.shared:
            self.kernel = self.add_weight(
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                name="kernel_all",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

            if self.use_bias:
                self.bias = self.add_weight(
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    name="bias_pos",
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )
            else:
                self.bias = None

        self.built = True

    def share_weights(self, layer: Layer) -> None:
        if not self.shared:
            return
        self.kernel = layer.kernel
        self.bias = layer.bias

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        """computing the perturbation analysis of the operator without the activation function

        Args:
            inputs: list of input tensors
            **kwargs

        Returns:
            List of updated tensors
        """
        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called on a list of inputs.")

        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        def conv_pos(x: tf.Tensor) -> tf.Tensor:
            return K.conv2d(
                x,
                K.maximum(z_value, self.kernel),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        def conv_neg(x: tf.Tensor) -> tf.Tensor:
            return K.conv2d(
                x,
                K.minimum(z_value, self.kernel),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        if self.dc_decomp:
            h_out = conv_pos(h) + conv_neg(g)
            g_out = conv_pos(g) + conv_neg(h)
        else:
            h_out, g_out = empty_tensor, empty_tensor

        if self.ibp:
            u_c_out = conv_pos(u_c) + conv_neg(l_c)
            l_c_out = conv_pos(l_c) + conv_neg(u_c)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            y = K.conv2d(
                b_u,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

            if len(w_u.shape) == len(b_u.shape):
                identity_tensor = self.diag_op(z_value * Flatten()(b_u[0][None]) + o_value)

                identity_tensor = K.reshape(identity_tensor, [-1] + list(b_u.shape[1:]))

                w_u_out = K.conv2d(
                    identity_tensor,
                    self.kernel,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
                w_u_out = K.expand_dims(w_u_out, 0) + z_value * K.expand_dims(y, 1)
                w_l_out = w_u_out
                b_u_out = 0 * y
                b_l_out = 0 * y

            else:
                # check for linearity
                x_max = self.perturbation_domain.get_upper(x, w_u - w_l, b_u - b_l)
                mask_b = o_value - K.sign(x_max)

                def step_pos(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return conv_pos(x), []

                def step_neg(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return conv_neg(x), []

                b_u_out = conv_pos(b_u) + conv_neg(b_l)
                b_l_out = conv_pos(b_l) + conv_neg(b_u)

                w_u_out = (
                    K.rnn(step_function=step_pos, inputs=w_u, initial_states=[], unroll=False)[1]
                    + K.rnn(step_function=step_neg, inputs=w_l, initial_states=[], unroll=False)[1]
                )
                w_l_out = (
                    K.rnn(step_function=step_pos, inputs=w_l, initial_states=[], unroll=False)[1]
                    + K.rnn(step_function=step_neg, inputs=w_u, initial_states=[], unroll=False)[1]
                )
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.use_bias:
            if self.dc_decomp:
                g_out = K.bias_add(g_out, K.minimum(z_value, self.bias), data_format=self.data_format)
                h_out = K.bias_add(h_out, K.maximum(z_value, self.bias), data_format=self.data_format)
            if self.affine:
                b_u_out = K.bias_add(b_u_out, self.bias, data_format=self.data_format)
                b_l_out = K.bias_add(b_l_out, self.bias, data_format=self.data_format)
            if self.ibp:
                u_c_out = K.bias_add(u_c_out, self.bias, data_format=self.data_format)
                l_c_out = K.bias_add(l_c_out, self.bias, data_format=self.data_format)

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        """
        Args:
            input_shape

        Returns:

        """
        assert len(input_shape) == self.nb_tensors

        if self.mode == ForwardMode.IBP:
            y_shape = input_shape[0]
        elif self.mode == ForwardMode.AFFINE:
            x_0_shape = input_shape[0]
            y_shape = input_shape[2]
        elif self.mode == ForwardMode.HYBRID:
            x_0_shape = input_shape[0]
            y_shape = input_shape[1]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.data_format == "channels_last":
            space = y_shape[1:-1]
        elif self.data_format == "channels_first":
            space = y_shape[2:]
        else:
            raise ValueError(f"Unknown data_format {self.data_format}")

        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i],
            )
            new_space.append(new_dim)
        if self.data_format == "channels_last":
            output_shape_keras = (y_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == "channels_first":
            output_shape_keras = (y_shape[0], self.filters) + tuple(new_space)
        else:
            raise ValueError(f"Unknown data_format {self.data_format}")

        if self.mode == ForwardMode.IBP:
            output_shape = [output_shape_keras] * 2
        else:
            input_dim = x_0_shape[-1]
            w_shape = tuple([output_shape_keras[0], input_dim] + list(output_shape_keras)[1:])
            if self.mode == ForwardMode.AFFINE:
                output_shape = [x_0_shape] + [w_shape, output_shape_keras] * 2
            elif self.mode == ForwardMode.HYBRID:
                output_shape = [x_0_shape] + [output_shape_keras, w_shape, output_shape_keras] * 2
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output_shape += [output_shape_keras] * 2

        return output_shape

    @property
    def keras_weights_names(self) -> List[str]:
        """Weights names of the corresponding Keras layer.

        Will be used to decide which weight to take from the keras layer in `reset_layer()`

        """
        weight_names = ["kernel"]
        if self.use_bias:
            weight_names.append("bias")
        return weight_names

    def freeze_weights(self) -> None:
        if not self.frozen_weights:
            self._trainable_weights = []
            self.frozen_weights = True

    def unfreeze_weights(self) -> None:
        if self.frozen_weights:
            if self.use_bias:
                self._trainable_weights = [self.bias] + self._trainable_weights
            self._trainable_weights = [self.kernel] + self._trainable_weights
            self.frozen_weights = False


class DecomonDense(DecomonLayer, Dense):
    """Forward LiRPA implementation of Dense layers.
    See Keras official documentation for further details on the Dense operator
    """

    original_keras_layer_class = Dense

    def __init__(
        self,
        units: int,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        kwargs.pop("activation", None)
        kwargs.pop("kernel_constraint", None)
        super().__init__(
            units=units,
            kernel_constraint=None,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        self.input_spec = [InputSpec(min_ndim=2) for _ in range(self.nb_tensors)]
        self.input_shape_build: Optional[List[tf.TensorShape]] = None
        self.op_dot = K.dot

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """

        assert len(input_shape) >= self.nb_tensors

        input_dim = input_shape[-1][-1]

        if not self.shared:
            self.kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=Project_initializer_pos(self.kernel_initializer),
                name="kernel_pos",
                regularizer=self.kernel_regularizer,
            )

            if self.use_bias:
                self.bias = self.add_weight(
                    shape=(self.units,),
                    initializer=self.bias_initializer,
                    name="bias_pos",
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                )
            else:
                self.bias = None

        # 6 inputs tensors :  h, g, x_min, x_max, W_u, b_u, W_l, b_l
        if self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=2),  # x_0
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # u_c
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # W_u
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # b_u
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # l_c
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # W_l
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # b_l
            ]
        elif self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # u_c
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # l_c
            ]
        elif self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=2),  # x_0
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # W_u
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # b_u
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # W_l
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # b_l
            ]
        if self.dc_decomp:
            self.input_spec += [
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # h
                InputSpec(min_ndim=2, axes={-1: input_dim}),
            ]  # g

        if self.has_backward_bounds:
            self.input_spec += [InputSpec(ndim=3, axes={-1: self.units, -2: self.units})]

        self.built = True
        self.input_shape_build = input_shape

    def share_weights(self, layer: Layer) -> None:
        if not self.shared:
            return
        self.kernel = layer.kernel
        if self.use_bias:
            self.bias = layer.bias

    def set_back_bounds(self, has_backward_bounds: bool) -> None:
        # check for activation
        if self.activation is not None and self.activation_name != "linear" and has_backward_bounds:
            raise ValueError()
        self.has_backward_bounds = has_backward_bounds
        if self.built and has_backward_bounds:
            if self.input_shape_build is None:
                raise ValueError("self.input_shape_build should not be None when calling set_back_bounds")
            # rebuild with an additional input
            self.build(self.input_shape_build)
        if self.has_backward_bounds:
            op = Dot(1)
            self.op_dot = lambda x, y: op([x, y])

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        """
        Args:
            inputs

        Returns:

        """
        if self.kernel is None:
            raise RuntimeError("self.kernel cannot be None when calling call()")

        z_value = K.cast(0.0, self.dtype)

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called " "on a list of inputs.")

        if self.has_backward_bounds:
            back_bound = inputs[-1]
            inputs = inputs[:-1]
            kernel = K.sum(self.kernel[None, :, :, None] * back_bound[:, None], 2)
            kernel_pos_back = K.maximum(z_value, kernel)
            kernel_neg_back = K.minimum(z_value, kernel)

        kernel_pos = K.maximum(z_value, self.kernel)
        kernel_neg = K.minimum(z_value, self.kernel)

        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        if self.dc_decomp:
            h_out = K.dot(h, kernel_pos) + K.dot(g, kernel_neg)
            g_out = K.dot(g, kernel_pos) + K.dot(h, kernel_neg)
        else:
            h_out, g_out = empty_tensor, empty_tensor

        if self.ibp:
            if not self.has_backward_bounds:
                u_c_out = self.op_dot(u_c, kernel_pos) + self.op_dot(l_c, kernel_neg)
                l_c_out = self.op_dot(l_c, kernel_pos) + self.op_dot(u_c, kernel_neg)
            else:
                u_c_out = self.op_dot(u_c, kernel_pos_back) + self.op_dot(l_c, kernel_neg_back)
                l_c_out = self.op_dot(l_c, kernel_pos_back) + self.op_dot(u_c, kernel_neg_back)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            if len(w_u.shape) == len(b_u.shape):
                y = K.dot(b_u, self.kernel)
                w_u_out = K.expand_dims(0 * y, 1) + K.expand_dims(self.kernel, 0)
                w_l_out = w_u_out
                b_u_out = z_value * y
                b_l_out = b_u_out

            if len(w_u.shape) != len(b_u.shape):
                b_u_out = K.dot(b_u, kernel_pos) + K.dot(b_l, kernel_neg)
                b_l_out = K.dot(b_l, kernel_pos) + K.dot(b_u, kernel_neg)

                if not self.has_backward_bounds:
                    w_u_out = K.dot(w_u, kernel_pos) + K.dot(w_l, kernel_neg)
                    w_l_out = K.dot(w_l, kernel_pos) + K.dot(w_u, kernel_neg)
                else:
                    raise NotImplementedError()  # bug somewhere
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.use_bias:
            if not self.has_backward_bounds:
                if self.ibp:
                    u_c_out = K.bias_add(u_c_out, self.bias, data_format="channels_last")
                    l_c_out = K.bias_add(l_c_out, self.bias, data_format="channels_last")
                if self.affine:
                    b_u_out = K.bias_add(b_u_out, self.bias, data_format="channels_last")
                    b_l_out = K.bias_add(b_l_out, self.bias, data_format="channels_last")
            else:
                b = K.sum(back_bound * self.bias[None, None], 1)
                if self.ibp:
                    u_c_out = u_c_out + b
                    l_c_out = l_c_out + b
                if self.affine:
                    b_u_out = b_u_out + b
                    b_l_out = b_l_out + b

            if self.dc_decomp:
                if self.has_backward_bounds:
                    raise NotImplementedError()
                h_out = K.bias_add(h_out, K.maximum(z_value, self.bias), data_format="channels_last")
                g_out = K.bias_add(g_out, K.minimum(z_value, self.bias), data_format="channels_last")

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        """
        Args:
            input_shape

        Returns:

        """

        assert len(input_shape) == self.nb_tensors
        output_shape = [list(elem) for elem in input_shape[: self.nb_tensors]]

        for i in range(1, self.nb_tensors):
            output_shape[i][-1] = self.units

        if self.mode == ForwardMode.IBP:
            output_shape[0][-1] = self.units

        return [tf.TensorShape(shape) for shape in output_shape]

    @property
    def keras_weights_names(self) -> List[str]:
        """Weights names of the corresponding Keras layer.

        Will be used to decide which weight to take from the keras layer in `reset_layer()`

        """
        weight_names = ["kernel"]
        if self.use_bias:
            weight_names.append("bias")
        return weight_names

    def freeze_weights(self) -> None:
        if not self.frozen_weights:
            self._trainable_weights = []
            self.frozen_weights = True

    def unfreeze_weights(self) -> None:
        if self.frozen_weights:
            if self.use_bias:
                self._trainable_weights = [self.bias] + self._trainable_weights
            self._trainable_weights = [self.kernel] + self._trainable_weights
            self.frozen_weights = False


class DecomonActivation(DecomonLayer, Activation):
    """Forward LiRPA implementation of Activation layers.
    See Keras official documentation for further details on the Activation operator
    """

    original_keras_layer_class = Activation

    def __init__(
        self,
        activation: str,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        slope: Union[str, Slope] = Slope.V_SLOPE,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            activation=activation,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        self.slope = Slope(slope)
        self.supports_masking = True
        self.activation = activations.get(activation)
        self.activation_name = activation

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        if self.finetune and self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            shape = input_shape[-1][1:]

            if self.activation_name != "linear" and self.mode != ForwardMode.IBP:
                if self.activation_name[:4] != "relu":
                    self.beta_u_f = self.add_weight(
                        shape=shape,
                        initializer="ones",
                        name="beta_u_f",
                        regularizer=None,
                        constraint=ClipAlpha(),
                    )
                self.beta_l_f = self.add_weight(
                    shape=shape,
                    initializer="ones",
                    name="beta_l_f",
                    regularizer=None,
                    constraint=ClipAlpha(),
                )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        if self.finetune and self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID] and self.activation_name != "linear":
            if self.activation_name[:4] == "relu":
                return self.activation(
                    inputs,
                    mode=self.mode,
                    dc_decomp=self.dc_decomp,
                    perturbation_domain=self.perturbation_domain,
                    finetune=self.beta_l_f,
                    slope=self.slope,
                )
            else:
                return self.activation(
                    inputs,
                    mode=self.mode,
                    dc_decomp=self.dc_decomp,
                    perturbation_domain=self.perturbation_domain,
                    finetune=[self.beta_u_f, self.beta_l_f],
                    slope=self.slope,
                )
        else:
            output = self.activation(
                inputs,
                mode=self.mode,
                perturbation_domain=self.perturbation_domain,
                dc_decomp=self.dc_decomp,
                slope=self.slope,
            )
            return output

    def reset_finetuning(self) -> None:
        if self.finetune and self.mode != ForwardMode.IBP:
            if self.activation_name != "linear":
                if self.activation_name[:4] == "relu":
                    K.set_value(self.beta_l_f, np.ones_like(self.beta_l_f.value()))
                else:
                    K.set_value(self.beta_u_f, np.ones_like(self.beta_u_f.value()))
                    K.set_value(self.beta_l_f, np.ones_like(self.beta_l_f.value()))

    def freeze_alpha(self) -> None:
        if not self.frozen_alpha:
            if self.finetune and self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                self._trainable_weights = []
            self.frozen_alpha = True

    def unfreeze_alpha(self) -> None:
        if self.frozen_alpha:
            if self.finetune and self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                if self.activation_name != "linear":
                    if self.activation_name[:4] != "relu":
                        self._trainable_weights += [self.beta_u_f, self.beta_l_f]
                    else:
                        self._trainable_weights += [self.beta_l_f]
            self.frozen_alpha = False


class DecomonFlatten(DecomonLayer, Flatten):
    """Forward LiRPA implementation of Flatten layers.
    See Keras official documentation for further details on the Flatten operator
    """

    original_keras_layer_class = Flatten

    def __init__(
        self,
        data_format: Optional[str] = None,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            data_format
            **kwargs
        """
        super().__init__(
            data_format=data_format,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        if self.mode == ForwardMode.HYBRID:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u
                InputSpec(min_ndim=2),  # w_u
                InputSpec(min_ndim=2),  # b_u
                InputSpec(min_ndim=1),  # l
                InputSpec(min_ndim=2),  # w_l
                InputSpec(min_ndim=2),  # b_l
            ]
        elif self.mode == ForwardMode.IBP:
            self.input_spec = [
                InputSpec(min_ndim=1),  # u
                InputSpec(min_ndim=1),  # l
            ]
        elif self.mode == ForwardMode.AFFINE:
            self.input_spec = [
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=2),  # w_u
                InputSpec(min_ndim=2),  # b_u
                InputSpec(min_ndim=2),  # w_l
                InputSpec(min_ndim=2),  # b_l
            ]
        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=1), InputSpec(min_ndim=1)]

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            self
            input_shape

        Returns:

        """
        return None

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        def op(x: tf.Tensor) -> tf.Tensor:
            return Flatten.call(self, x)

        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        if self.dc_decomp:
            h_out = op(h)
            g_out = op(g)
        else:
            h_out, g_out = empty_tensor, empty_tensor

        if self.ibp:
            u_c_out = op(u_c)
            l_c_out = op(l_c)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_out = op(b_u)
            b_l_out = op(b_l)
            input_dim = x.shape[-1]
            output_shape = np.prod(list(K.int_shape(b_u_out))[1:])
            w_u_out = K.reshape(w_u, (-1, input_dim, output_shape))
            w_l_out = K.reshape(w_l, (-1, input_dim, output_shape))
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )


class DecomonBatchNormalization(DecomonLayer, BatchNormalization):
    """Forward LiRPA implementation of Batch Normalization layers.
    See Keras official documentation for further details on the BatchNormalization operator
    """

    original_keras_layer_class = BatchNormalization

    def __init__(
        self,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
        moving_mean_initializer: str = "zeros",
        moving_variance_initializer: str = "ones",
        beta_regularizer: Optional[str] = None,
        gamma_regularizer: Optional[str] = None,
        beta_constraint: Optional[str] = None,
        gamma_constraint: Optional[str] = None,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        super().build(input_shape)
        self.input_spec = [InputSpec(min_ndim=len(elem)) for elem in input_shape]

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        output_shape: tf.TensorShape = BatchNormalization.compute_output_shape(self, input_shape[-1])

        if self.mode == ForwardMode.IBP:
            output = [output_shape] * 2

        elif self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            x_shape = input_shape[0]
            input_dim = x_shape[-1]
            w_shape = np.array(output_shape)[:, None]
            w_shape[:, 0] = input_dim
            if self.mode == ForwardMode.AFFINE:
                output = [x_shape] + [w_shape, output_shape] * 2
            else:
                output = [x_shape] + [output_shape, w_shape, output_shape] * 2
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [output_shape, output_shape]
        return output

    def call(self, inputs: List[tf.Tensor], training: Optional[bool] = None, **kwargs: Any) -> List[tf.Tensor]:
        if training is None:
            training = K.learning_phase()

        z_value = K.cast(0.0, self.dtype)

        if training:
            raise NotImplementedError("not working during training")

        def call_op(x: tf.Tensor, training: bool) -> tf.Tensor:
            return BatchNormalization.call(self, x, training=training)

        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        y = call_op(inputs[-1], training=training)

        n_dim = len(y.shape)
        shape = [1] * n_dim
        for i, ax in enumerate(self.axis):
            shape[ax] = self.moving_mean.shape[i]

        if self.gamma is None:  # scale = False
            gamma = tf.ones(shape)
        else:  # scale = True
            gamma = K.reshape(self.gamma + z_value, shape)
        if self.beta is None:  # center = False
            beta = tf.zeros(shape)
        else:  # center = True
            beta = K.reshape(self.beta + z_value, shape)
        moving_mean = K.reshape(self.moving_mean + z_value, shape)
        moving_variance = K.reshape(self.moving_variance + z_value, shape)

        if self.ibp:
            u_c_0 = (u_c - moving_mean) / K.sqrt(moving_variance + self.epsilon)
            l_c_0 = (l_c - moving_mean) / K.sqrt(moving_variance + self.epsilon)
            u_c_out = K.maximum(z_value, gamma) * u_c_0 + K.minimum(z_value, gamma) * l_c_0 + beta
            l_c_out = K.maximum(z_value, gamma) * l_c_0 + K.minimum(z_value, gamma) * u_c_0 + beta
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_0 = (b_u - moving_mean) / K.sqrt(moving_variance + self.epsilon)
            b_l_0 = (b_l - moving_mean) / K.sqrt(moving_variance + self.epsilon)

            b_u_out = K.maximum(z_value, gamma) * b_u_0 + K.minimum(z_value, gamma) * b_l_0 + beta
            b_l_out = K.maximum(z_value, gamma) * b_l_0 + K.minimum(z_value, gamma) * b_u_0 + beta

            gamma = K.expand_dims(gamma, 1)
            moving_variance = K.expand_dims(moving_variance, 1)

            w_u_0 = w_u / K.sqrt(moving_variance + self.epsilon)
            w_l_0 = w_l / K.sqrt(moving_variance + self.epsilon)
            w_u_out = K.maximum(z_value, gamma) * w_u_0 + K.minimum(z_value, gamma) * w_l_0
            w_l_out = K.maximum(z_value, gamma) * w_l_0 + K.minimum(z_value, gamma) * w_u_0
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.dc_decomp:
            raise NotImplementedError()
        else:
            h_out, g_out = empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )

    @property
    def keras_weights_names(self) -> List[str]:
        """Weights names of the corresponding Keras layer.

        Will be used to decide which weight to take from the keras layer in `reset_layer()`

        """
        weight_names = ["moving_mean", "moving_variance"]
        if self.center:
            weight_names.append("beta")
        if self.scale:
            weight_names.append("gamma")
        return weight_names


class DecomonDropout(DecomonLayer, Dropout):
    """Forward LiRPA implementation of Dropout layers.
    See Keras official documentation for further details on the Dropout operator
    """

    original_keras_layer_class = Dropout

    def __init__(
        self,
        rate: float,
        noise_shape: Optional[Tuple[int, ...]] = None,
        seed: Optional[int] = None,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            rate=rate,
            noise_shape=noise_shape,
            seed=seed,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        return input_shape

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        super().build(input_shape)
        self.input_spec = [InputSpec(min_ndim=len(elem)) for elem in input_shape]

    def call(self, inputs: List[tf.Tensor], training: Optional[bool] = None, **kwargs: Any) -> List[tf.Tensor]:
        if training is None:
            training = K.learning_phase()

        if training:
            raise NotImplementedError("not working during training")

        return inputs


class DecomonInputLayer(DecomonLayer, InputLayer):
    """Forward LiRPA implementation of Dropout layers.
    See Keras official documentation for further details on the Dropout operator
    """

    original_keras_layer_class = InputLayer

    def __init__(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[str] = None,
        input_tensor: Optional[tf.Tensor] = None,
        sparse: Optional[bool] = None,
        name: Optional[str] = None,
        ragged: Optional[bool] = None,
        type_spec: Optional[tf.TypeSpec] = None,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        if type_spec is not None:
            super().__init__(
                input_shape=input_shape,
                batch_size=batch_size,
                dtype=dtype,
                type_spec=type_spec,
                input_tensor=input_tensor,
                sparse=sparse,
                name=name,
                ragged=ragged,
                perturbation_domain=perturbation_domain,
                dc_decomp=dc_decomp,
                mode=mode,
                finetune=finetune,
                shared=shared,
                fast=fast,
                **kwargs,
            )
        else:
            super().__init__(
                input_shape=input_shape,
                batch_size=batch_size,
                dtype=dtype,
                input_tensor=input_tensor,
                sparse=sparse,
                name=name,
                ragged=ragged,
                mode=mode,
                **kwargs,
            )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        return inputs

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        return input_shape
