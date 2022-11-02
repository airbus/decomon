from __future__ import absolute_import
import tensorflow as tf
from .core import DecomonLayer
import tensorflow.keras.backend as K
from tensorflow.keras.backend import bias_add, conv2d
import numpy as np
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import initializers

# from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Activation,
    Flatten,
    Reshape,
    Dot,
    Input,
    BatchNormalization,
    Dropout,
    Lambda,
    InputSpec,
    InputLayer
)

try:
    from keras.layers.merge import _Merge as Merge
except ModuleNotFoundError:
    from tensorflow.python.keras.layers.merge import _Merge as Merge


class DecomonConv2D(Conv2D, DecomonLayer):
    """
    Forward LiRPA implementation of Conv2d layers.
    See Keras official documentation for further details on the Conv2d operator

    """

    def __init__(self, filters, kernel_size, mode=F_HYBRID.name, **kwargs):

        activation = kwargs["activation"]
        if "activation" in kwargs:
            kwargs["activation"] = None
        super(DecomonConv2D, self).__init__(filters=filters, kernel_size=kernel_size, mode=mode, **kwargs)
        self.kernel_constraint_pos_ = NonNeg()
        self.kernel_constraint_neg_ = NonPos()

        self.kernel_pos = None
        self.kernel_neg = None
        self.kernel = None
        self.kernel_constraints_pos = None
        self.kernel_constraints_neg = None
        self.activation = activations.get(activation)
        self.bias = None
        self.w_ = None

        if self.mode == F_HYBRID.name:
            self.input_spec = [
                # InputSpec(min_ndim=4),  # y
                InputSpec(min_ndim=2),  # z
                InputSpec(min_ndim=4),  # u
                InputSpec(min_ndim=4),  # wu
                InputSpec(min_ndim=4),  # bu
                InputSpec(min_ndim=4),  # l
                InputSpec(min_ndim=4),  # wl
                InputSpec(min_ndim=4),  # bl
            ]
        elif self.mode == F_IBP.name:
            self.input_spec = [
                # InputSpec(min_ndim=4),  # y
                # InputSpec(min_ndim=2),  # z
                InputSpec(min_ndim=4),  # u
                InputSpec(min_ndim=4),  # l
            ]
        elif self.mode == F_FORWARD.name:
            self.input_spec = [
                # InputSpec(min_ndim=4),  # y
                InputSpec(min_ndim=2),  # z
                InputSpec(min_ndim=4),  # wu
                InputSpec(min_ndim=4),  # bu
                InputSpec(min_ndim=4),  # wl
                InputSpec(min_ndim=4),  # bl
            ]
        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=4), InputSpec(min_ndim=4)]

        self.diag_op = Lambda(lambda x: tf.linalg.diag(x))

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
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

        if self.finetune and self.mode == F_HYBRID.name:
            # create extra parameters that can be optimized


            self.alpha_ = self.add_weight(
                shape=kernel_shape,
                initializer="ones",
                name="alpha_f",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.gamma_ = self.add_weight(
                shape=kernel_shape,
                initializer="ones",
                name="gamma_f",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            n_in_ = input_shape[-1][1:]
            self.alpha_pos_in = self.add_weight(
                shape=n_in_,
                initializer="ones",
                name="alpha_f_pos_in",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.alpha_neg_in = self.add_weight(
                shape=n_in_,
                initializer="ones",
                name="alpha_f_neg_in",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.gamma_pos_in = self.add_weight(
                shape=n_in_,
                initializer="ones",
                name="gamma_f_pos_in",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.gamma_neg_in = self.add_weight(
                shape=n_in_,
                initializer="ones",
                name="gamma_f_neg_in",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            n_out_ = self.compute_output_shape(input_shape)[-1][1:]
            self.alpha_pos_out = self.add_weight(
                shape=n_out_,
                initializer="ones",
                name="alpha_f_pos_out",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.alpha_neg_out = self.add_weight(
                shape=n_out_,
                initializer="ones",
                name="alpha_f_neg_out",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.gamma_pos_out = self.add_weight(
                shape=n_out_,
                initializer="ones",
                name="gamma_f_pos_out",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            self.gamma_neg_out = self.add_weight(
                shape=n_out_,
                initializer="ones",
                name="gamma_f_neg_out",
                regularizer=None,
                constraint=ClipAlpha(),
            )

            """
            dim_alpha = tuple(input_shape[0][1:])
            # create extra parameters that can be optimized

            self.alpha_u_f = self.add_weight(
                shape=dim_alpha, initializer="ones", name="alpha_u_f", regularizer=None, constraint=ClipAlpha()
            )
            self.alpha_l_f = self.add_weight(
                shape=dim_alpha, initializer="ones", name="alpha_l_f", regularizer=None, constraint=ClipAlpha()
            )
            """

        """
        if self.mode == F_HYBRID.name:
            # inputs tensors: h, g, x_min, x_max, W_u, b_u, W_l, b_l
            if channel_axis == -1:
                self.input_spec = [
                    #InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                    InputSpec(min_ndim=2),  # x
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # u_c
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # w_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # l_c
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # w_l
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_l
                ]
            else:
                self.input_spec = [
                    #InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                    InputSpec(min_ndim=2),  # x
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # u_c
                    InputSpec(min_ndim=4, axes={channel_axis + 1: input_dim}),  # w_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # l_c
                    InputSpec(min_ndim=4, axes={channel_axis + 1: input_dim}),  # w_l
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_l
                ]
        elif self.mode == F_IBP.name:
            self.input_spec = [
                # InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                # InputSpec(min_ndim=2),  # x
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # u_c
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # l_c
            ]
        elif self.mode == F_FORWARD.name:
            # inputs tensors: h, g, x_min, x_max, W_u, b_u, W_l, b_l
            if channel_axis == -1:
                self.input_spec = [
                    #InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                    InputSpec(min_ndim=2),  # x
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # w_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # w_l
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_l
                ]
            else:
                self.input_spec = [
                    #InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                    InputSpec(min_ndim=2),  # x
                    InputSpec(min_ndim=4, axes={channel_axis + 1: input_dim}),  # w_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_u
                    InputSpec(min_ndim=4, axes={channel_axis + 1: input_dim}),  # w_l
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_l
                ]

        if self.dc_decomp:
            self.input_spec += [
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # h
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # g
            ]

        """
        self.built = True

    def get_backward_weights(self, inputs, flatten=True):

        # if self.w_ is None:
        z_value = K.cast(0.0, K.floatx())
        o_value = K.cast(1., K.floatx())
        b_u = inputs[-1]
        n_in = np.prod(b_u.shape[1:])

        id_ = self.diag_op(z_value * Flatten()(b_u[0][None]) + o_value)

        id_ = K.reshape(id_, [-1] + list(b_u.shape[1:]))

        w_ = conv2d(
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
        n_repeat = int(w_.shape[-1] / self.bias.shape[-1])
        b_ = K.reshape(K.repeat(self.bias[None], n_repeat), (-1,))
        return w_, b_

    def shared_weights(self, layer):
        if not self.shared:
            pass
        self.kernel = layer.kernel
        self.bias = layer.bias

    def call_linear(self, inputs, **kwargs):
        """
        computing the perturbation analysis of the operator without the activation function
        :param inputs: list of input tensors
        :param kwargs:
        :return: List of updated tensors
        """
        z_value = K.cast(0.0, K.floatx())
        o_value = K.cast(1., K.floatx())

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called " "on a list of inputs.")

        if self.mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
            raise ValueError("unknown  forward mode {}".format(self.mode))

        if self.mode == F_HYBRID.name:
            if self.dc_decomp:
                # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs[: self.nb_tensors]
                x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
                x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif self.mode == F_IBP.name:
            if self.dc_decomp:
                # y, x_0, u_c, l_c, h, g = inputs[: self.nb_tensors]
                u_c, l_c, h, g = inputs[: self.nb_tensors]
            else:
                # y, x_0, u_c, l_c = inputs[: self.nb_tensors]
                u_c, l_c = inputs[: self.nb_tensors]
        elif self.mode == F_FORWARD.name:
            if self.dc_decomp:
                # y, x_0, w_u, b_u, w_l, b_l, h, g = inputs[: self.nb_tensors]
                x_0, w_u, b_u, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                # y, x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]
                x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]

        # y_ = conv2d(
        #    y,
        #    self.kernel,
        #    strides=self.strides,
        #    padding=self.padding,
        #    data_format=self.data_format,
        #    dilation_rate=self.dilation_rate,
        # )

        def conv_pos(x):
            return conv2d(
                x,
                K.maximum(z_value, self.kernel),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        def conv_neg(x):
            return conv2d(
                x,
                K.minimum(z_value, self.kernel),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        if self.finetune and self.mode == F_HYBRID.name:
            def conv_pos_alpha(x):
                return conv2d(
                    x,
                    K.maximum(z_value, self.kernel * self.alpha_),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )

            def conv_pos_gamma(x):
                return conv2d(
                    x,
                    K.maximum(z_value, self.kernel * self.gamma_),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )

            def conv_neg_alpha(x):
                return conv2d(
                    x,
                    K.minimum(z_value, self.kernel * self.alpha_),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )

            def conv_neg_gamma(x):
                return conv2d(
                    x,
                    K.minimum(z_value, self.kernel * self.gamma_),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )

        if self.dc_decomp:
            h_ = conv_pos(h) + conv_neg(g)
            g_ = conv_pos(g) + conv_neg(h)

        if self.mode in [F_HYBRID.name, F_IBP.name]:
            u_c_ = conv_pos(u_c) + conv_neg(l_c)
            l_c_ = conv_pos(l_c) + conv_neg(u_c)

        if self.mode in [F_FORWARD.name, F_HYBRID.name]:

            y_ = conv2d(
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

                w_u_ = conv2d(
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

                if self.finetune and self.mode == F_HYBRID.name:
                    self.frozen_alpha = True
                    self.finetune = False
                    self._trainable_weights = self._trainable_weights[:-10]

            else:
                # check for linearity
                # mask_b = 0 * y_
                # mask_b = 0*b_u
                # if self.mode in [F_HYBRID.name]:  # F_FORWARD.name
                x_max = get_upper(x_0, w_u - w_l, b_u - b_l, self.convex_domain)
                mask_b = o_value - K.sign(x_max)
                mask_a = o_value - mask_b

                if self.mode == F_HYBRID.name and self.finetune:

                    b_u_ = self.alpha_pos_out[None] * conv_pos_alpha(self.alpha_pos_in[None] * (b_u - u_c)) + \
                           (o_value - self.alpha_pos_out[None]) * conv_pos(u_c) + \
                           self.alpha_neg_out[None] * conv_neg_alpha(self.alpha_neg_in[None] * (b_l - l_c)) + \
                           (o_value - self.alpha_neg_out[None]) * conv_neg(l_c)

                    b_l_ = self.gamma_pos_out[None] * conv_pos_gamma(self.gamma_pos_in[None] * (b_l - l_c)) + \
                           (o_value - self.gamma_pos_out[None]) * conv_pos(l_c) + \
                           self.gamma_neg_out[None] * conv_neg_gamma(self.gamma_neg_in[None] * (b_u - u_c)) + \
                           (o_value - self.gamma_neg_out[None]) * conv_neg(u_c)

                    """
                    b_u_ = self.alpha_pos_out[None]*conv_pos_alpha(self.alpha_pos_in[None]*(b_u-u_c)) +\
                           self.alpha_neg_out[None]*conv_neg_alpha(self.alpha_neg_in[None]*mask_a*(b_l-l_c) +
                                          self.alpha_pos_in[None]*mask_b*(b_u-u_c)) +\
                           (o_value-self)*conv_pos((o_value-self.alpha_pos_in[None])*u_c) +\
                           conv_neg((o_value-self.alpha_neg_in[None])*mask_a*l_c +
                                    (o_value - self.alpha_pos_in[None])*mask_b*u_c)



                    b_l_ = conv_pos_gamma(self.gamma_pos_in[None]*(b_l-l_c)) +\
                           conv_neg_gamma(self.gamma_neg_in[None]*mask_a*(b_u-u_c) +
                                          self.gamma_pos_in[None]*mask_b*(b_l-l_c)) +\
                           conv_pos((o_value-self.gamma_pos_in[None])*l_c) +\
                           conv_neg((o_value-self.gamma_neg_in[None])*mask_a*u_c +
                                    (o_value-self.gamma_pos_in[None])*mask_b*l_c)
                    """

                    """
                    b_u_ = conv_pos(self.alpha_u_f * b_u + (1 - self.alpha_u_f) * u_c) + conv_neg(
                        mask_a * (self.alpha_l_f * b_l + (1 - self.alpha_l_f) * l_c)
                        + mask_b * (self.alpha_u_f * b_u + (1 - self.alpha_u_f) * u_c)
                    )

                    b_l_ = conv_pos(self.alpha_l_f * b_l + (1 - self.alpha_l_f) * l_c) + conv_neg(
                        mask_a * (self.alpha_u_f * b_u + (1 - self.alpha_u_f) * u_c)
                        + mask_b * (self.alpha_l_f * b_l + (1 - self.alpha_l_f) * l_c)
                    )
                    """
                else:
                    """
                    b_u_ = conv_pos(b_u) + conv_neg(mask_a * b_l + mask_b * b_u)

                    b_l_ = conv_pos(b_l) + conv_neg(mask_a * b_u + mask_b * b_l)
                    """
                    b_u_ = conv_pos(b_u) + conv_neg(b_l)

                    b_l_ = conv_pos(b_l) + conv_neg(b_u)

                mask_a = K.expand_dims(mask_a, 1)
                mask_b = K.expand_dims(mask_b, 1)

                def step_pos(x, _):
                    return conv_pos(x), []

                def step_neg(x, _):
                    return conv_neg(x), []

                if self.mode == F_HYBRID.name and self.finetune:

                    def step_pos_alpha(x, _):
                        return conv_pos_alpha(x), []

                    def step_neg_alpha(x, _):
                        return conv_neg_alpha(x), []

                    def step_pos_gamma(x, _):
                        return conv_pos_gamma(x), []

                    def step_neg_gamma(x, _):
                        return conv_neg_gamma(x), []

                    w_u_ = (
                            self.alpha_pos_out[None, None] * K.rnn(step_function=step_pos_alpha,
                                                                   inputs=self.alpha_pos_in[None, None] * w_u,
                                                                   initial_states=[], unroll=False)[1]
                            + self.alpha_neg_out * K.rnn(step_function=step_neg_alpha,
                                                         inputs=self.alpha_neg_in[None, None] * w_l, initial_states=[],
                                                         unroll=False)[1])

                    w_l_ = (
                            self.gamma_pos_out[None, None] * K.rnn(step_function=step_pos_gamma,
                                                                   inputs=self.gamma_pos_in[None, None] * w_l,
                                                                   initial_states=[], unroll=False)[1]
                            + self.gamma_neg_out[None, None] * K.rnn(step_function=step_neg_gamma,
                                                                     inputs=self.gamma_neg_in[None, None] * w_u,
                                                                     initial_states=[],
                                                                     unroll=False)[1])

                    """
                    alpha_u_ = self.alpha_u_f[None, None, :]
                    alpha_l_ = self.alpha_l_f[None, None, :]

                    w_u_ = (
                        K.rnn(step_function=step_pos, inputs=alpha_u_ * w_u, initial_states=[], unroll=False)[1]
                        + K.rnn(
                            step_function=step_neg,
                            inputs=mask_a * alpha_l_ * w_l + mask_b * alpha_u_ * w_u,
                            initial_states=[],
                            unroll=False,
                        )[1]
                    )
                    w_l_ = (
                        K.rnn(step_function=step_pos, inputs=alpha_l_ * w_l, initial_states=[], unroll=False)[1]
                        + K.rnn(
                            step_function=step_neg,
                            inputs=mask_a * alpha_u_ * w_u + mask_b * alpha_l_ * w_l,
                            initial_states=[],
                            unroll=False,
                        )[1]
                    )
                    """
                else:

                    """
                    w_u_ = (
                        K.rnn(step_function=step_pos, inputs=w_u, initial_states=[], unroll=False)[1]
                        + K.rnn(
                            step_function=step_neg, inputs=mask_a * w_l + mask_b * w_u, initial_states=[], unroll=False
                        )[1]
                    )
                    w_l_ = (
                        K.rnn(step_function=step_pos, inputs=w_l, initial_states=[], unroll=False)[1]
                        + K.rnn(
                            step_function=step_neg, inputs=mask_a * w_u + mask_b * w_l, initial_states=[], unroll=False
                        )[1]
                    )
                    """
                    w_u_ = (
                            K.rnn(step_function=step_pos, inputs=w_u, initial_states=[], unroll=False)[1]
                            + K.rnn(
                        step_function=step_neg, inputs=w_l, initial_states=[], unroll=False
                    )[1]
                    )
                    w_l_ = (
                            K.rnn(step_function=step_pos, inputs=w_l, initial_states=[], unroll=False)[1]
                            + K.rnn(
                        step_function=step_neg, inputs=w_u, initial_states=[], unroll=False
                    )[1]
                    )

        # add bias
        if self.use_bias:
            # y_ = bias_add(y_, self.bias, data_format=self.data_format)
            if self.dc_decomp:
                g_ = bias_add(g_, K.minimum(z_value, self.bias), data_format=self.data_format)
                h_ = bias_add(h_, K.maximum(z_value, self.bias), data_format=self.data_format)

            if self.mode in [F_HYBRID.name, F_FORWARD.name]:
                b_u_ = bias_add(b_u_, self.bias, data_format=self.data_format)
                b_l_ = bias_add(b_l_, self.bias, data_format=self.data_format)
            if self.mode in [F_HYBRID.name, F_IBP.name]:
                u_c_ = bias_add(u_c_, self.bias, data_format=self.data_format)
                l_c_ = bias_add(l_c_, self.bias, data_format=self.data_format)

        if self.mode == F_HYBRID.name:
            upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
            u_c_ = K.minimum(upper_, u_c_)
            lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)
            l_c_ = K.maximum(lower_, l_c_)

        if self.mode == F_HYBRID.name:
            # output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        elif self.mode == F_IBP.name:
            # output = [y_, x_0, u_c_, l_c_]
            output = [u_c_, l_c_]
        elif self.mode == F_FORWARD.name:
            # output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
            output = [x_0, w_u_, b_u_, w_l_, b_l_]

        if self.dc_decomp:
            output += [h_, g_]

        return output

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :return:

        """
        output = self.call_linear(inputs, **kwargs)

        # temporary fix until all activations are ready
        if self.activation is not None:
            output = self.activation(output, dc_decomp=self.dc_decomp, mode=self.mode)

        return output

    def compute_output_shape(self, input_shape):
        """

        :param input_shape:
        :return:

        """
        assert len(input_shape) == self.nb_tensors

        if self.mode == F_IBP.name:
            y_shape = input_shape[0]

        if self.mode == F_FORWARD.name:
            x_0_shape = input_shape[0]
            y_shape = input_shape[2]

        if self.mode == F_HYBRID.name:
            x_0_shape = input_shape[0]
            y_shape = input_shape[1]

        # y_shape, x_0_shape = input_shape[:2]

        if self.data_format == "channels_last":
            space = y_shape[1:-1]
        elif self.data_format == "channels_first":
            space = y_shape[2:]

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

        # b_u_shape_, b_l_shape_, u_c_shape, l_c_shape = [output_shape] * 4
        # input_dim = x_0_shape[-1]

        if self.mode == F_IBP.name:
            # output_shape_ = [output_shape, x_0_shape, output_shape, output_shape]
            output_shape_ = [output_shape] * 2
        else:
            input_dim = x_0_shape[-1]
            w_shape_ = tuple([output_shape[0], input_dim] + list(output_shape)[1:])

        if self.mode == F_FORWARD.name:
            # output_shape_ = [output_shape, x_0_shape] + [w_shape_, output_shape] * 2
            output_shape_ = [x_0_shape] + [w_shape_, output_shape] * 2

        if self.mode == F_HYBRID.name:
            # output_shape_ = [output_shape, x_0_shape] + [output_shape, w_shape_, output_shape] * 2
            output_shape_ = [x_0_shape] + [output_shape, w_shape_, output_shape] * 2

        if self.dc_decomp:
            output_shape_ += [output_shape] * 2

        return output_shape_

    def reset_layer(self, layer):
        """

        :param layer:
        :return:

        """
        # assert than we have the same configuration
        assert isinstance(layer, Conv2D), "wrong type of layers..."
        params = layer.get_weights()
        if self.finetune:
            params += self.get_weights()[2:]
        self.set_weights(params)

    def freeze_weights(self):

        if not self.frozen_weights:

            if self.finetune and self.mode == F_HYBRID.name:
                if self.use_bias:
                    self._trainable_weights = self._trainable_weights[2:]
                else:
                    self._trainable_weights = self._trainable_weights[1:]
            else:
                self._trainable_weights = []

            self.frozen_weights = True

    def unfreeze_weights(self):

        if self.frozen_weights:

            if self.use_bias:
                self._trainable_weights = [self.bias] + self._trainable_weights
            self._trainable_weights = [self.kernel] + self._trainable_weights
            self.frozen_weights = False

    def freeze_alpha(self):
        if not self.frozen_alpha:
            if self.finetune and self.mode == F_HYBRID.name:
                self._trainable_weights = self._trainable_weights[:-10]
                self.frozen_alpha = True

    def unfreeze_alpha(self):
        if self.frozen_alpha:
            if self.finetune and self.mode == F_HYBRID.name:
                self._trainable_weights += [self.alpha_, self.gamma_, self.alpha_pos_in, self.alpha_pos_out,
                                            self.gamma_pos_in, self.gamma_pos_out,
                                            self.alpha_pos_out, self.alpha_neg_out,
                                            self.gamma_pos_out, self.gamma_neg_out]
            self.frozen_alpha = False

    def reset_finetuning(self):
        if self.finetune and self.mode == F_HYBRID.name:
            K.set_value(self.alpha_, np.ones_like(self.alpha_.value()))
            K.set_value(self.gamma_, np.ones_like(self.gamma_.value()))
            K.set_value(self.alpha_in, np.ones_like(self.alpha_pos_in.value()))
            K.set_value(self.alpha_in, np.ones_like(self.alpha_pos_out.value()))
            K.set_value(self.gamma_in, np.ones_like(self.gamma_pos_in.value()))
            K.set_value(self.gamma_in, np.ones_like(self.gamma_pos_out.value()))
            K.set_value(self.alpha_out, np.ones_like(self.alpha_pos_out.value()))
            K.set_value(self.alpha_out, np.ones_like(self.alpha_neg_out.value()))
            K.set_value(self.gamma_out, np.ones_like(self.gamma_pos_out.value()))
            K.set_value(self.gamma_out, np.ones_like(self.gamma_pos_in.value()))
