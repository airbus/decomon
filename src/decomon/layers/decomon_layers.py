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
from tensorflow.python.keras.utils.generic_utils import to_list

from decomon.layers import activations
from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.utils import (
    ClipAlpha,
    ClipAlphaAndSumtoOne,
    NonPos,
    Project_initializer_pos,
)
from decomon.utils import ConvexDomainType, Slope, get_lower, get_upper


class DecomonConv2D(Conv2D, DecomonLayer):
    """Forward LiRPA implementation of Conv2d layers.
    See Keras official documentation for further details on the Conv2d operator

    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        self.kernel_constraint_pos_ = NonNeg()
        self.kernel_constraint_neg_ = NonPos()

        self.kernel_pos = None
        self.kernel_neg = None
        self.kernel = None
        self.kernel_constraints_pos = None
        self.kernel_constraints_neg = None
        self.bias = None
        self.w_ = None

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

        if self.finetune and self.mode == ForwardMode.HYBRID:
            # create extra parameters that can be optimized

            n_in_ = input_shape[-1][1:]
            self.n_in_ = [e for e in n_in_]
            nb_comp = int(np.prod(to_list(self.kernel_size)) * self.filters / np.prod(to_list(self.strides)))
            self.n_comp = nb_comp

            self.alpha_ = self.add_weight(
                shape=[nb_comp] + n_in_,
                initializer="ones",
                name="alpha_f",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.gamma_ = self.add_weight(
                shape=nb_comp + n_in_,
                initializer="ones",
                name="gamma_f",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            n_out_ = self.compute_output_shape(input_shape)[-1][1:]
            self.n_out_ = [e for e in n_out_]

            self.alpha_out = self.add_weight(
                shape=[nb_comp] + self.n_out_,
                initializer="ones",
                name="alpha_f",
                regularizer=None,
                constraint=ClipAlphaAndSumtoOne(),
            )

            self.gamma_out = self.add_weight(
                shape=[nb_comp] + self.n_out_,
                initializer="ones",
                name="alpha_f",
                regularizer=None,
                constraint=ClipAlphaAndSumtoOne(),
            )

        self.built = True

    def get_backward_weights(self, inputs: List[tf.Tensor], flatten: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:

        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)

        b_u = inputs[-1]
        n_in = np.prod(b_u.shape[1:])
        id_ = K.cast(self.diag_op(z_value * Flatten(dtype=self.dtype)(b_u[0][None]) + o_value), self.dtype)
        id_ = K.reshape(id_, [-1] + list(b_u.shape[1:]))
        w_ = K.conv2d(
            id_,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )
        if flatten:
            if self.data_format == "channels_last":
                c_in, height, width, c_out = w_.shape
                w_ = K.reshape(w_, (c_in * height * width, c_out))
            else:
                c_out, height, width, c_in = w_.shape
                w_ = K.reshape(w_, (c_out, c_in * height * width))
                w_ = K.transpose(w_, (1, 0))

        w_ = K.reshape(w_, (n_in, -1))
        if self.use_bias:
            if self.bias is None:
                raise RuntimeError("self.bias cannot be None when calling get_backward_weights()")
            n_repeat = int(w_.shape[-1] / self.bias.shape[-1])
            b_ = K.reshape(K.repeat(self.bias[None], n_repeat), (-1,))
        else:
            b_ = K.cast(0.0, self.dtype) * w_[1][None]
        return w_, b_

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

        if self.mode == ForwardMode.HYBRID:
            if self.dc_decomp:
                x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif self.mode == ForwardMode.IBP:
            if self.dc_decomp:
                u_c, l_c, h, g = inputs[: self.nb_tensors]
            else:
                u_c, l_c = inputs[: self.nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            if self.dc_decomp:
                x_0, w_u, b_u, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]
        else:
            raise ValueError(f"unknown  forward mode {self.mode}")

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
            h_ = conv_pos(h) + conv_neg(g)
            g_ = conv_pos(g) + conv_neg(h)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
            u_c_ = conv_pos(u_c) + conv_neg(l_c)
            l_c_ = conv_pos(l_c) + conv_neg(u_c)

        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:

            y_ = K.conv2d(
                b_u,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

            if len(w_u.shape) == len(b_u.shape):
                id_ = self.diag_op(z_value * Flatten()(b_u[0][None]) + o_value)

                id_ = K.reshape(id_, [-1] + list(b_u.shape[1:]))

                w_u_ = K.conv2d(
                    id_,
                    self.kernel,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
                w_u_ = K.expand_dims(w_u_, 0) + z_value * K.expand_dims(y_, 1)
                w_l_ = w_u_
                b_u_ = 0 * y_
                b_l_ = 0 * y_

                if self.finetune and self.mode == ForwardMode.HYBRID:
                    self.frozen_alpha = True
                    self.finetune = False
                    self._trainable_weights = self._trainable_weights[:-4]

            else:
                # check for linearity
                x_max = get_upper(x_0, w_u - w_l, b_u - b_l, self.convex_domain)
                mask_b = o_value - K.sign(x_max)
                mask_a = o_value - mask_b

                def step_pos(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return conv_pos(x), []

                def step_neg(x: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
                    return conv_neg(x), []

                if self.mode == ForwardMode.HYBRID and self.finetune:

                    b_u_ = (
                        K.rnn(
                            step_function=step_pos,
                            inputs=self.alpha_[None] * (b_u - u_c)[:, None],
                            initial_states=[],
                            unroll=False,
                        )[1]
                        + u_c_[:, None]
                        + K.rnn(
                            step_function=step_neg,
                            inputs=self.alpha_[None] * (b_l - l_c)[:, None],
                            initial_states=[],
                            unroll=False,
                        )[1]
                    )

                    b_l_ = (
                        K.rnn(
                            step_function=step_pos,
                            inputs=self.gamma_[None] * (b_l - l_c)[:, None],
                            initial_states=[],
                            unroll=False,
                        )[1]
                        + l_c_[:, None]
                        + K.rnn(
                            step_function=step_neg,
                            inputs=self.gamma_[None] * (b_u - u_c)[:, None],
                            initial_states=[],
                            unroll=False,
                        )[1]
                    )

                else:
                    b_u_ = conv_pos(b_u) + conv_neg(b_l)
                    b_l_ = conv_pos(b_l) + conv_neg(b_u)

                if self.mode == ForwardMode.HYBRID and self.finetune:
                    n_x = w_u.shape[1]

                    w_u_alpha = K.reshape(
                        self.alpha_[None, None] * w_u[:, :, None], [-1, n_x * self.n_comp] + self.n_in_
                    )
                    w_l_alpha = K.reshape(
                        self.alpha_[None, None] * w_l[:, :, None], [-1, n_x * self.n_comp] + self.n_in_
                    )
                    w_u_gamma = K.reshape(
                        self.gamma_[None, None] * w_u[:, :, None], [-1, n_x * self.n_comp] + self.n_in_
                    )
                    w_l_gamma = K.reshape(
                        self.gamma_[None, None] * w_l[:, :, None], [-1, n_x * self.n_comp] + self.n_in_
                    )

                    w_u_ = (
                        K.rnn(step_function=step_pos, inputs=w_u_alpha, initial_states=[], unroll=False)[1]
                        + K.rnn(step_function=step_neg, inputs=w_l_alpha, initial_states=[], unroll=False)[1]
                    )
                    w_l_ = (
                        K.rnn(step_function=step_pos, inputs=w_l_gamma, initial_states=[], unroll=False)[1]
                        + K.rnn(step_function=step_neg, inputs=w_u_gamma, initial_states=[], unroll=False)[1]
                    )

                    n_out = [e for e in w_u_.shape[2:]]
                    w_u_ = K.reshape(w_u_, [-1, n_x, self.n_comp] + n_out)
                    w_l_ = K.reshape(w_l_, [-1, n_x, self.n_comp] + n_out)

                else:

                    w_u_ = (
                        K.rnn(step_function=step_pos, inputs=w_u, initial_states=[], unroll=False)[1]
                        + K.rnn(step_function=step_neg, inputs=w_l, initial_states=[], unroll=False)[1]
                    )
                    w_l_ = (
                        K.rnn(step_function=step_pos, inputs=w_l, initial_states=[], unroll=False)[1]
                        + K.rnn(step_function=step_neg, inputs=w_u, initial_states=[], unroll=False)[1]
                    )

        # add bias
        if self.mode == ForwardMode.HYBRID:
            upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
            lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)

            if self.finetune:
                # retrieve the best relaxation of the n_comp possible

                upper_ = K.min(upper_, 1)
                lower_ = K.max(lower_, 1)

                # affine combination on the output: take the best

                b_u_ = K.sum(self.alpha_out[None] * b_u_, 1)
                b_l_ = K.sum(self.gamma_out[None] * b_l_, 1)
                w_u_ = K.sum(self.alpha_out[None, None] * w_u_, 2)
                w_l_ = K.sum(self.gamma_out[None, None] * w_l_, 2)

            u_c_ = K.minimum(upper_, u_c_)

            l_c_ = K.maximum(lower_, l_c_)

        if self.use_bias:
            if self.dc_decomp:
                g_ = K.bias_add(g_, K.minimum(z_value, self.bias), data_format=self.data_format)
                h_ = K.bias_add(h_, K.maximum(z_value, self.bias), data_format=self.data_format)

            if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
                b_u_ = K.bias_add(b_u_, self.bias, data_format=self.data_format)
                b_l_ = K.bias_add(b_l_, self.bias, data_format=self.data_format)
            if self.mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
                u_c_ = K.bias_add(u_c_, self.bias, data_format=self.data_format)
                l_c_ = K.bias_add(l_c_, self.bias, data_format=self.data_format)

        if self.mode == ForwardMode.HYBRID:
            upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
            u_c_ = K.minimum(upper_, u_c_)
            lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)
            l_c_ = K.maximum(lower_, l_c_)

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        elif self.mode == ForwardMode.IBP:
            output = [u_c_, l_c_]
        elif self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_, b_u_, w_l_, b_l_]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [h_, g_]

        return output

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
            output_shape = (y_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == "channels_first":
            output_shape = (y_shape[0], self.filters) + tuple(new_space)
        else:
            raise ValueError(f"Unknown data_format {self.data_format}")

        if self.mode == ForwardMode.IBP:
            output_shape_ = [output_shape] * 2
        else:
            input_dim = x_0_shape[-1]
            w_shape_ = tuple([output_shape[0], input_dim] + list(output_shape)[1:])
            if self.mode == ForwardMode.AFFINE:
                output_shape_ = [x_0_shape] + [w_shape_, output_shape] * 2
            elif self.mode == ForwardMode.HYBRID:
                output_shape_ = [x_0_shape] + [output_shape, w_shape_, output_shape] * 2
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output_shape_ += [output_shape] * 2

        return output_shape_

    def reset_layer(self, layer: Layer) -> None:
        """
        Args:
            layer

        Returns:

        """
        # assert than we have the same configuration
        assert isinstance(layer, Conv2D), "wrong type of layers..."
        params = layer.get_weights()
        if self.finetune:
            params += self.get_weights()[2:]
        self.set_weights(params)

    def freeze_weights(self) -> None:

        if not self.frozen_weights:

            if self.finetune and self.mode == ForwardMode.HYBRID:
                if self.use_bias:
                    self._trainable_weights = self._trainable_weights[2:]
                else:
                    self._trainable_weights = self._trainable_weights[1:]
            else:
                self._trainable_weights = []

            self.frozen_weights = True

    def unfreeze_weights(self) -> None:

        if self.frozen_weights:

            if self.use_bias:
                self._trainable_weights = [self.bias] + self._trainable_weights
            self._trainable_weights = [self.kernel] + self._trainable_weights
            self.frozen_weights = False

    def freeze_alpha(self) -> None:
        if not self.frozen_alpha:
            if self.finetune and self.mode == ForwardMode.HYBRID:
                self._trainable_weights = self._trainable_weights[:-4]
                self.frozen_alpha = True

    def unfreeze_alpha(self) -> None:
        if self.frozen_alpha:
            if self.finetune and self.mode == ForwardMode.HYBRID:
                self._trainable_weights += [self.alpha_, self.gamma_, self.alpha_out, self.gamma_out]
            self.frozen_alpha = False

    def reset_finetuning(self) -> None:
        if self.finetune and self.mode == ForwardMode.HYBRID:

            K.set_value(self.alpha_, np.ones_like(self.alpha_.value()))
            K.set_value(self.gamma_, np.ones_like(self.gamma_.value()))
            K.set_value(self.alpha_out, np.ones_like(self.alpha_neg_out.value()))
            K.set_value(self.gamma_out, np.ones_like(self.gamma_pos_out.value()))


class DecomonDense(Dense, DecomonLayer):
    """Forward LiRPA implementation of Dense layers.
    See Keras official documentation for further details on the Dense operator
    """

    def __init__(
        self,
        units: int,
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        self.kernel_constraint_pos_ = NonNeg()
        self.kernel_constraint_neg_ = NonPos()
        self.input_spec = [InputSpec(min_ndim=2) for _ in range(self.nb_tensors)]
        self.kernel_pos = None
        self.kernel_neg = None
        self.kernel: Optional[tf.Variable] = None
        self.kernel_constraints_pos = None
        self.kernel_constraints_neg = None
        self.dot_op = Dot(axes=(1, 2))
        self.n_subgrad = 0  # deprecated optimization scheme
        self.input_shape_build: Optional[List[tf.TensorShape]] = None
        self.op_dot = K.dot

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """

        assert len(input_shape) >= self.nb_tensors

        if self.mode == ForwardMode.IBP:
            input_dim = input_shape[0][-1]
        elif self.mode == ForwardMode.HYBRID:
            input_dim = input_shape[1][-1]
        elif self.mode == ForwardMode.AFFINE:
            input_dim = input_shape[2][-1]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if not self.shared:
            self.kernel = self.add_weight(
                shape=(input_dim, self.units),
                initializer=Project_initializer_pos(self.kernel_initializer),
                name="kernel_pos",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraints_pos,
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

        if self.finetune and self.mode == ForwardMode.HYBRID:
            # create extra parameters that can be optimized
            self.alpha_ = self.add_weight(
                shape=(input_dim, self.units),
                initializer="ones",
                name="alpha_",
                regularizer=None,
                constraint=ClipAlpha(),
            )
            self.gamma_ = self.add_weight(
                shape=(input_dim, self.units),
                initializer="ones",
                name="gamma_",
                regularizer=None,
                constraint=ClipAlpha(),
            )

        # False
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
            op_ = Dot(1)
            self.op_dot = lambda x, y: op_([x, y])

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
            kernel_ = K.sum(self.kernel[None, :, :, None] * back_bound[:, None], 2)
            kernel_pos_back = K.maximum(z_value, kernel_)
            kernel_neg_back = K.minimum(z_value, kernel_)

        kernel_pos = K.maximum(z_value, self.kernel)
        kernel_neg = K.minimum(z_value, self.kernel)

        if self.finetune and self.mode == ForwardMode.HYBRID:

            kernel_pos_alpha = K.maximum(z_value, self.kernel * self.alpha_)
            kernel_pos_gamma = K.maximum(z_value, self.kernel * self.gamma_)
            kernel_neg_alpha = K.minimum(z_value, self.kernel * self.alpha_)
            kernel_neg_gamma = K.minimum(z_value, self.kernel * self.gamma_)

        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = K.dot(h, kernel_pos) + K.dot(g, kernel_neg)
            g_ = K.dot(g, kernel_pos) + K.dot(h, kernel_neg)
            rest = 2
        else:
            rest = 0
        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors - rest]
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[: self.nb_tensors - rest]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors - rest]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
            if not self.linear_layer:
                if not self.has_backward_bounds:
                    u_c_ = self.op_dot(u_c, kernel_pos) + self.op_dot(l_c, kernel_neg)
                    l_c_ = self.op_dot(l_c, kernel_pos) + self.op_dot(u_c, kernel_neg)
                else:
                    u_c_ = self.op_dot(u_c, kernel_pos_back) + self.op_dot(l_c, kernel_neg_back)
                    l_c_ = self.op_dot(l_c, kernel_pos_back) + self.op_dot(u_c, kernel_neg_back)

            else:

                # check convex_domain
                if len(self.convex_domain) and self.convex_domain["name"] == ConvexDomainType.BALL:

                    if self.mode == ForwardMode.IBP:
                        x_0 = (u_c + l_c) / 2.0
                    b_ = (0 * self.kernel[0])[None]
                    if self.has_backward_bounds:
                        raise NotImplementedError()
                    u_c_ = get_upper(x_0, self.kernel[None], b_, convex_domain=self.convex_domain)
                    l_c_ = get_lower(x_0, self.kernel[None], b_, convex_domain=self.convex_domain)

                else:
                    if not self.has_backward_bounds:
                        u_c_ = self.op_dot(u_c, kernel_pos) + self.op_dot(l_c, kernel_neg)
                        l_c_ = self.op_dot(l_c, kernel_pos) + self.op_dot(u_c, kernel_neg)
                    else:
                        u_c_ = self.op_dot(u_c, kernel_pos_back) + self.op_dot(l_c, kernel_neg_back)
                        l_c_ = self.op_dot(l_c, kernel_pos_back) + self.op_dot(u_c, kernel_neg_back)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:

            if len(w_u.shape) == len(b_u.shape):
                y_ = K.dot(b_u, self.kernel)
                w_u_ = K.expand_dims(0 * y_, 1) + K.expand_dims(self.kernel, 0)
                w_l_ = w_u_
                b_u_ = z_value * y_
                b_l_ = b_u_
                if self.finetune and self.mode == ForwardMode.HYBRID:
                    self.frozen_alpha = True
                    self._trainable_weights = self._trainable_weights[:-2]  # not optimal

            if len(w_u.shape) != len(b_u.shape):
                # first layer, it is necessary linear
                if self.finetune and self.mode == ForwardMode.HYBRID:
                    b_u_ = (
                        K.dot(b_u - u_c, kernel_pos_alpha)
                        + K.dot((b_l - l_c), kernel_neg_alpha)
                        + K.dot(u_c, kernel_pos)
                        + K.dot(l_c, kernel_neg)
                    )  # focus....

                    b_l_ = (
                        K.dot(b_l - l_c, kernel_pos_gamma)
                        + K.dot((b_u - u_c), kernel_neg_gamma)
                        + K.dot(l_c, kernel_pos)
                        + K.dot(u_c, kernel_neg)
                    )

                else:
                    b_u_ = K.dot(b_u, kernel_pos) + K.dot(b_l, kernel_neg)
                    b_l_ = K.dot(b_l, kernel_pos) + K.dot(b_u, kernel_neg)

                if self.finetune and self.mode == ForwardMode.HYBRID:

                    if self.has_backward_bounds:
                        raise ValueError("last layer should not be finetuned")

                    w_u_ = K.dot(w_u, kernel_pos_alpha) + K.dot(w_l, kernel_neg_alpha)
                    w_l_ = K.dot(w_l, kernel_pos_gamma) + K.dot(w_u, kernel_neg_gamma)

                else:

                    if not self.has_backward_bounds:
                        w_u_ = K.dot(w_u, kernel_pos) + K.dot(w_l, kernel_neg)
                        w_l_ = K.dot(w_l, kernel_pos) + K.dot(w_u, kernel_neg)
                    else:

                        raise NotImplementedError()  # bug somewhere

        if self.use_bias:

            if not self.has_backward_bounds:
                if self.mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
                    u_c_ = K.bias_add(u_c_, self.bias, data_format="channels_last")
                    l_c_ = K.bias_add(l_c_, self.bias, data_format="channels_last")
                if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                    b_u_ = K.bias_add(b_u_, self.bias, data_format="channels_last")
                    b_l_ = K.bias_add(b_l_, self.bias, data_format="channels_last")
            else:
                b_ = K.sum(back_bound * self.bias[None, None], 1)
                if self.mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
                    u_c_ = u_c_ + b_
                    l_c_ = l_c_ + b_
                if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                    b_u_ = b_u_ + b_
                    b_l_ = b_l_ + b_

            if self.dc_decomp:
                if self.has_backward_bounds:
                    raise NotImplementedError()
                h_ = K.bias_add(h_, K.maximum(z_value, self.bias), data_format="channels_last")
                g_ = K.bias_add(g_, K.minimum(z_value, self.bias), data_format="channels_last")

        if self.mode == ForwardMode.HYBRID:
            upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
            lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)

            l_c_ = K.maximum(lower_, l_c_)
            u_c_ = K.minimum(upper_, u_c_)

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_, b_u_, w_l_, b_l_]
        if self.mode == ForwardMode.IBP:
            output = [u_c_, l_c_]

        if self.dc_decomp:
            output += [h_, g_]

        return output

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

    def reset_layer(self, dense: Layer) -> None:
        """
        Args:
            dense

        Returns:

        """
        # assert than we have the same configuration
        assert isinstance(dense, Dense), "wrong type of layers..."
        if dense.built:

            params = dense.get_weights()
            if self.finetune:
                params += self.get_weights()[2:]
            self.set_weights(params)
        else:
            raise ValueError(f"the layer {dense.name} has not been built yet")

    def freeze_weights(self) -> None:

        if not self.frozen_weights:

            if self.finetune and self.mode == ForwardMode.HYBRID:
                if self.use_bias:
                    self._trainable_weights = self._trainable_weights[2:]
                else:
                    self._trainable_weights = self._trainable_weights[1:]
            else:
                self._trainable_weights = []

            self.frozen_weights = True

    def unfreeze_weights(self) -> None:

        if self.frozen_weights:

            if self.use_bias:
                self._trainable_weights = [self.bias] + self._trainable_weights
            self._trainable_weights = [self.kernel] + self._trainable_weights
            self.frozen_weights = False

    def freeze_alpha(self) -> None:
        if not self.frozen_alpha:
            if self.finetune and self.mode == ForwardMode.HYBRID:
                self._trainable_weights = self._trainable_weights[:-2]
            self.frozen_alpha = True

    def unfreeze_alpha(self) -> None:
        if self.frozen_alpha:
            if self.finetune and self.mode == ForwardMode.HYBRID:
                self._trainable_weights += [self.alpha_, self.gamma_]
            self.frozen_alpha = False

    def reset_finetuning(self) -> None:
        if self.finetune and self.mode == ForwardMode.HYBRID:
            K.set_value(self.alpha_, np.ones_like(self.alpha_.value()))
            K.set_value(self.gamma_, np.ones_like(self.gamma_.value()))


class DecomonActivation(Activation, DecomonLayer):
    """Forward LiRPA implementation of Activation layers.
    See Keras official documentation for further details on the Activation operator
    """

    def __init__(
        self,
        activation: str,
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
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
                    convex_domain=self.convex_domain,
                    finetune=self.beta_l_f,
                    slope=self.slope,
                )
            else:
                return self.activation(
                    inputs,
                    mode=self.mode,
                    dc_decomp=self.dc_decomp,
                    convex_domain=self.convex_domain,
                    finetune=[self.beta_u_f, self.beta_l_f],
                    slope=self.slope,
                )
        else:
            output = self.activation(
                inputs, mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp, slope=self.slope
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


class DecomonFlatten(Flatten, DecomonLayer):
    """Forward LiRPA implementation of Flatten layers.
    See Keras official documentation for further details on the Flatten operator
    """

    def __init__(
        self,
        data_format: Optional[str] = None,
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
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

        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = op(h)
            g_ = op(g)
        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[: self.nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
            u_c_ = op(u_c)
            l_c_ = op(l_c)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            b_u_ = op(b_u)
            b_l_ = op(b_l)

            output_shape = np.prod(list(K.int_shape(b_u_))[1:])
            input_dim = K.int_shape(x_0)[-1]

            w_u_ = K.reshape(w_u, (-1, input_dim, output_shape))
            w_l_ = K.reshape(w_l, (-1, input_dim, output_shape))

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        elif self.mode == ForwardMode.IBP:
            output = [u_c_, l_c_]
        elif self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_, b_u_, w_l_, b_l_]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [h_, g_]

        return output


class DecomonBatchNormalization(BatchNormalization, DecomonLayer):
    """Forward LiRPA implementation of Batch Normalization layers.
    See Keras official documentation for further details on the BatchNormalization operator
    """

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
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        BatchNormalization.build(self, input_shape[0])

        self.input_spec = [InputSpec(min_ndim=len(elem)) for elem in input_shape]

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:

        output_shape_: tf.TensorShape = BatchNormalization.compute_output_shape(self, input_shape[-1])

        if self.mode == ForwardMode.IBP:
            output = [output_shape_] * 2

        elif self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            x_shape = input_shape[0]
            input_dim = x_shape[-1]
            w_shape = np.array(output_shape_)[:, None]
            w_shape[:, 0] = input_dim
            if self.mode == ForwardMode.AFFINE:
                output = [x_shape] + [w_shape, output_shape_] * 2
            else:
                output = [x_shape] + [output_shape_, w_shape, output_shape_] * 2
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            output += [output_shape_, output_shape_]
        return output

    def call(self, inputs: List[tf.Tensor], training: Optional[bool] = None, **kwargs: Any) -> List[tf.Tensor]:

        if training is None:
            training = K.learning_phase()

        z_value = K.cast(0.0, self.dtype)

        if training:
            raise NotImplementedError("not working during training")

        def call_op(x: tf.Tensor, training: bool) -> tf.Tensor:
            return BatchNormalization.call(self, x, training=training)

        if self.dc_decomp:
            raise NotImplementedError()

        if self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif self.mode == ForwardMode.IBP:
            u_c, l_c = inputs[: self.nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]

        y_ = call_op(inputs[-1], training=training)

        n_dim = len(y_.shape)
        tuple_ = [1] * n_dim
        for i, ax in enumerate(self.axis):
            tuple_[ax] = self.moving_mean.shape[i]

        gamma_ = K.reshape(self.gamma + z_value, tuple_)
        beta_ = K.reshape(self.beta + z_value, tuple_)
        moving_mean_ = K.reshape(self.moving_mean + z_value, tuple_)
        moving_variance_ = K.reshape(self.moving_variance + z_value, tuple_)

        if self.mode in [ForwardMode.HYBRID, ForwardMode.IBP]:

            u_c_0 = (u_c - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)
            l_c_0 = (l_c - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)

            u_c_ = K.maximum(z_value, gamma_) * u_c_0 + K.minimum(z_value, gamma_) * l_c_0 + beta_
            l_c_ = K.maximum(z_value, gamma_) * l_c_0 + K.minimum(z_value, gamma_) * u_c_0 + beta_

        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:

            b_u_0 = (b_u - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)
            b_l_0 = (b_l - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)

            b_u_ = K.maximum(z_value, gamma_) * b_u_0 + K.minimum(z_value, gamma_) * b_l_0 + beta_
            b_l_ = K.maximum(z_value, gamma_) * b_l_0 + K.minimum(z_value, gamma_) * b_u_0 + beta_

            gamma_ = K.expand_dims(gamma_, 1)
            moving_variance_ = K.expand_dims(moving_variance_, 1)

            w_u_0 = w_u / K.sqrt(moving_variance_ + self.epsilon)
            w_l_0 = w_l / K.sqrt(moving_variance_ + self.epsilon)
            w_u_ = K.maximum(z_value, gamma_) * w_u_0 + K.minimum(z_value, gamma_) * w_l_0
            w_l_ = K.maximum(z_value, gamma_) * w_l_0 + K.minimum(z_value, gamma_) * w_u_0

        if self.mode == ForwardMode.HYBRID:
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == ForwardMode.IBP:
            output = [u_c_, l_c_]
        if self.mode == ForwardMode.AFFINE:
            output = [x_0, w_u_, b_u_, w_l_, b_l_]

        return output

    def reset_layer(self, layer: Layer) -> None:
        """
        Args:
            layer

        Returns:

        """
        assert isinstance(layer, BatchNormalization), "wrong type of layers..."
        params = layer.get_weights()
        self.set_weights(params)


class DecomonDropout(Dropout, DecomonLayer):
    """Forward LiRPA implementation of Dropout layers.
    See Keras official documentation for further details on the Dropout operator
    """

    def __init__(
        self,
        rate: float,
        noise_shape: Optional[Tuple[int, ...]] = None,
        seed: Optional[int] = None,
        convex_domain: Optional[Dict[str, Any]] = None,
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
            convex_domain=convex_domain,
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
        super().build(input_shape[0])
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
        convex_domain: Optional[Dict[str, Any]] = None,
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
                convex_domain=convex_domain,
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

    def get_linear(self) -> bool:
        return self.linear_layer
