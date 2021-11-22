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
)

try:
    from keras.layers.merge import _Merge as Merge
except ModuleNotFoundError:
    from tensorflow.python.keras.layers.merge import _Merge as Merge

from decomon.layers import activations
from decomon.layers.utils import NonPos, ClipAlpha, MultipleConstraint, Project_initializer_pos, Project_initializer_neg
from tensorflow.python.keras.utils import conv_utils
from decomon.layers.utils import get_upper, get_lower, sort
from .maxpooling import DecomonMaxPooling2D
from .decomon_reshape import DecomonReshape, DecomonPermute
from .decomon_merge_layers import (
    DecomonConcatenate,
    DecomonAverage,
    DecomonAdd,
    DecomonMinimum,
    DecomonMaximum,
    DecomonSubtract,
    to_monotonic_merge,
)
from .utils import grad_descent
from .core import F_FORWARD, F_IBP, F_HYBRID
import warnings
import inspect

try:
    from .deel_lip import DecomonGroupSort
except ModuleNotFoundError:
    pass


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

        if self.mode == F_HYBRID.name:
            self.input_spec = [
                InputSpec(min_ndim=4),  # y
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
                InputSpec(min_ndim=4),  # y
                InputSpec(min_ndim=2),  # z
                InputSpec(min_ndim=4),  # u
                InputSpec(min_ndim=4),  # l
            ]
        elif self.mode == F_FORWARD.name:
            self.input_spec = [
                InputSpec(min_ndim=4),  # y
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
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

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


        if self.mode == F_HYBRID.name:
            # inputs tensors: h, g, x_min, x_max, W_u, b_u, W_l, b_l
            if channel_axis == -1:
                self.input_spec = [
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
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
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
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
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                InputSpec(min_ndim=2),  # x
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # u_c
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # l_c
            ]
        elif self.mode == F_HYBRID.name:
            # inputs tensors: h, g, x_min, x_max, W_u, b_u, W_l, b_l
            if channel_axis == -1:
                self.input_spec = [
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                    InputSpec(min_ndim=2),  # x
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # w_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_u
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # w_l
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_l
                ]
            else:
                self.input_spec = [
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
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

        self.built = True

    def call_linear(self, inputs, **kwargs):
        """
        computing the perturbation analysis of the operator without the activation function
        :param inputs: list of input tensors
        :param kwargs:
        :return: List of updated tensors
        """

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called " "on a list of inputs.")

        if self.mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
            raise ValueError("unknown  forward mode {}".format(self.mode))

        if self.mode == F_HYBRID.name:
            if self.dc_decomp:
                y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif self.mode == F_IBP.name:
            if self.dc_decomp:
                y, x_0, u_c, l_c, h, g = inputs[: self.nb_tensors]
            else:
                y, x_0, u_c, l_c = inputs[: self.nb_tensors]
        elif self.mode == F_FORWARD.name:
            if self.dc_decomp:
                y, x_0, w_u, b_u, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                y, x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]

        y_ = conv2d(
            y,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        def conv_pos(x):
            return conv2d(
                x,
                K.maximum(0.0, self.kernel),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        def conv_neg(x):
            return conv2d(
                x,
                K.minimum(0.0, self.kernel),
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        if self.finetune:
            def conv_pos_alpha(x):
                return conv2d(
                    x,
                    K.maximum(0.0, self.kernel*self.alpha_),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
            def conv_pos_gamma(x):
                return conv2d(
                    x,
                    K.maximum(0.0, self.kernel*self.gamma_),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )

            def conv_neg_alpha(x):
                return conv2d(
                    x,
                    K.minimum(0.0, self.kernel*self.alpha_),
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
            def conv_neg_gamma(x):
                return conv2d(
                    x,
                    K.minimum(0.0, self.kernel*self.gamma_),
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
            if len(w_u.shape) == len(b_u.shape):
                id_ = self.diag_op(0 * Flatten()(y[0][None]) + 1.0)

                id_ = K.reshape(id_, [-1] + list(y.shape[1:]))

                w_u_ = conv2d(
                    id_,
                    self.kernel,
                    strides=self.strides,
                    padding=self.padding,
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate,
                )
                w_u_ = K.expand_dims(w_u_, 0) + 0 * K.expand_dims(y_, 1)
                w_l_ = w_u_
                b_u_ = 0 * y_
                b_l_ = 0 * y_

                if self.finetune and self.mode == F_HYBRID.name:
                    self.frozen_alpha = True
                    self._trainable_weights = self._trainable_weights[:-2]

            else:
                # check for linearity
                mask_b = 0 * y
                if self.mode in [F_HYBRID.name]:  # F_FORWARD.name
                    x_max = get_upper(x_0, w_u - w_l, b_u - b_l, self.convex_domain)
                    mask_b = 1.0 - K.sign(x_max)
                mask_a = 1.0 - mask_b

                if self.mode == F_HYBRID.name and self.finetune:

                    b_u_ = conv_pos_alpha(b_u-u_c) + conv_neg_alpha(mask_a*(b_l-l_c) + mask_b*(b_u-u_c)) +\
                           conv_pos(u_c) + conv_neg(mask_a*l_c + mask_b*u_c)

                    b_l_ = conv_pos_gamma(b_l-l_c) + conv_neg_gamma(mask_a*(b_u-u_c) + mask_b*(b_l-l_c)) +\
                           conv_pos(l_c) + conv_neg(mask_a*u_c + mask_b*l_c)
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
                    b_u_ = conv_pos(b_u) + conv_neg(mask_a * b_l + mask_b * b_u)

                    b_l_ = conv_pos(b_l) + conv_neg(mask_a * b_u + mask_b * b_l)

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
                            K.rnn(step_function=step_pos_alpha, inputs=w_u, initial_states=[], unroll=False)[1]
                          + K.rnn(step_function=step_neg_alpha,inputs=mask_a * w_l + mask_b * w_u, initial_states=[],
                            unroll=False)[1])

                    w_l_ = (
                            K.rnn(step_function=step_pos_gamma, inputs=w_l, initial_states=[], unroll=False)[1]
                            + K.rnn(step_function=step_neg_gamma, inputs=mask_a * w_u + mask_b * w_l, initial_states=[],
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

        # add bias
        if self.use_bias:
            y_ = bias_add(y_, self.bias, data_format=self.data_format)
            if self.dc_decomp:
                g_ = bias_add(g_, K.minimum(0.0, self.bias), data_format=self.data_format)
                h_ = bias_add(h_, K.maximum(0.0, self.bias), data_format=self.data_format)

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
            output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        elif self.mode == F_IBP.name:
            output = [y_, x_0, u_c_, l_c_]
        elif self.mode == F_FORWARD.name:
            output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

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
        y_shape, x_0_shape = input_shape[:2]

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
        input_dim = x_0_shape[-1]

        if self.mode == F_IBP.name:
            output_shape_ = [output_shape, x_0_shape, output_shape, output_shape]
        else:
            w_shape_ = tuple([output_shape[0], input_dim] + list(output_shape)[1:])

        if self.mode == F_FORWARD.name:
            output_shape_ = [output_shape, x_0_shape] + [w_shape_, output_shape] * 2

        if self.mode == F_HYBRID.name:
            output_shape_ = [output_shape, x_0_shape] + [output_shape, w_shape_, output_shape] * 2

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
                self._trainable_weights = self._trainable_weights[:-2]
                self.frozen_alpha = True

    def unfreeze_alpha(self):
        if not self.frozen_alpha:
            if self.finetune and self.mode == F_HYBRID.name:
                self._trainable_weights += [self.alpha_, self.gamma_]
            self.frozen_alpha = False

    def reset_finetuning(self):
        if self.finetune and self.mode == F_HYBRID.name:

            K.set_value(self.alpha_, np.ones_like(self.alpha_.value()))
            K.set_value(self.gamma_, np.ones_like(self.gamma_.value()))


class DecomonDense(Dense, DecomonLayer):
    """
    Forward LiRPA implementation of Dense layers.
    See Keras official documentation for further details on the Dense operator
    """

    def __init__(self, units, mode=F_HYBRID.name, **kwargs):
        if "activation" not in kwargs:
            kwargs["activation"] = None
        activation = kwargs["activation"]
        kwargs["units"] = units
        kwargs["kernel_constraint"] = None
        super(DecomonDense, self).__init__(mode=mode, **kwargs)
        self.mode = mode
        self.kernel_constraint_pos_ = NonNeg()
        self.kernel_constraint_neg_ = NonPos()
        self.input_spec = [InputSpec(min_ndim=2) for _ in range(self.nb_tensors)]
        self.kernel_pos = None
        self.kernel_neg = None
        self.kernel = None
        self.kernel_constraints_pos = None
        self.kernel_constraints_neg = None
        self.activation = activations.get(activation)
        self.dot_op = Dot(axes=(1, 2))
        self.n_subgrad = 0  # deprecated optimization scheme
        if activation is None:
            self.activation_name = 'linear'
        else:
            self.activation_name = activation

    def build(self, input_shape):
        """
        :param input_shape: list of input shape
        :return:
        """

        assert len(input_shape) >= self.nb_tensors

        # here pay attention to compute input_dim

        input_dim = input_shape[0][-1]

        input_x = input_shape[1][-1]

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

        if self.finetune and self.mode == F_HYBRID.name:
            # create extra parameters that can be optimized

            if self.activation_name!='linear':

                if self.activation_name[:4]!='relu':
                    self.beta_u_f_ = self.add_weight(
                        shape=(self.units,), initializer="ones", name="beta_u_f", regularizer=None, constraint=ClipAlpha()
                    )
                self.beta_l_f_ = self.add_weight(
                    shape=(self.units,), initializer="ones", name="beta_l_f", regularizer=None, constraint=ClipAlpha()
                )


            self.alpha_ = self.add_weight(
                shape=(input_dim, self.units), initializer="ones", name="alpha_", regularizer=None, constraint=ClipAlpha()
            )
            self.gamma_ = self.add_weight(
                shape=(input_dim, self.units), initializer="ones", name="gamma_", regularizer=None, constraint=ClipAlpha()
            )



        # False
        # 6 inputs tensors :  h, g, x_min, x_max, W_u, b_u, W_l, b_l
        if self.mode == F_HYBRID.name:
            self.input_spec = [
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # y
                InputSpec(min_ndim=2, axes={-1: input_x}),  # x_0
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # u_c
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # W_u
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # b_u
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # l_c
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # W_l
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # b_l
            ]
        elif self.mode == F_IBP.name:
            self.input_spec = [
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # y
                InputSpec(min_ndim=2, axes={-1: input_x}),  # x_0
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # u_c
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # l_c
            ]
        elif self.mode == F_FORWARD.name:
            self.input_spec = [
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # y
                InputSpec(min_ndim=2, axes={-1: input_x}),  # x_0
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

        self.built = True

    def call_linear(self, inputs):
        """

        :param inputs:
        :return:
        """

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called " "on a list of inputs.")

        kernel_pos = K.maximum(0.0, self.kernel)
        kernel_neg = K.minimum(0.0, self.kernel)

        if self.finetune:

            kernel_pos_alpha = K.maximum(0., self.kernel*self.alpha_)
            kernel_pos_gamma = K.maximum(0., self.kernel*self.gamma_)
            kernel_neg_alpha = K.minimum(0., self.kernel*self.alpha_)
            kernel_neg_gamma = K.minimum(0., self.kernel * self.gamma_)


        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = K.dot(h, kernel_pos) + K.dot(g, kernel_neg)
            g_ = K.dot(g, kernel_pos) + K.dot(h, kernel_neg)

        if self.mode == F_HYBRID.name:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]

        if self.mode == F_IBP.name:
            y, x_0, u_c, l_c = inputs[:4]
        if self.mode == F_FORWARD.name:
            y, x_0, w_u, b_u, w_l, b_l = inputs[:6]

        y_ = K.dot(y, self.kernel)  # + K.dot(y, self.kernel_neg)

        mask_b = 0 * (y)

        if self.mode == [F_HYBRID.name, F_FORWARD.name]:
            if len(w_u.shape) != len(b_u.shape):
                x_max = get_upper(x_0, w_u - w_l, b_u - b_l, self.convex_domain)
                mask_b = 1.0 - K.sign(x_max)

        mask_a = 1.0 - mask_b

        if self.mode in [F_HYBRID.name, F_IBP.name]:

            u_c_ = K.dot(u_c, kernel_pos) + K.dot(l_c, kernel_neg)
            l_c_ = K.dot(l_c, kernel_pos) + K.dot(u_c, kernel_neg)

        if self.mode in [F_HYBRID.name, F_FORWARD.name]:

            if len(w_u.shape) == len(b_u.shape):
                w_u_ = K.expand_dims(0 * (y_), 1) + K.expand_dims(self.kernel, 0)
                w_l_ = w_u_
                b_u_ = 0 * y_
                b_l_ = b_u_
                if self.finetune and self.mode == F_HYBRID.name:
                    self.frozen_alpha = True
                    self._trainable_weights = self._trainable_weights[:-2] # not optimal

            if len(w_u.shape) != len(b_u.shape):
                # first layer, it is necessary linear

                if self.finetune and self.mode == F_HYBRID.name:

                    b_u_ = K.dot(b_u-u_c, kernel_pos_alpha) +\
                           K.dot(mask_a * (b_l-l_c) + mask_b * (b_u-u_c), kernel_neg_alpha) +\
                           K.dot(u_c, kernel_pos) + K.dot(mask_a*l_c + mask_b*u_c, kernel_neg)

                    b_l_ = K.dot(b_l-l_c, kernel_pos_gamma) +\
                           K.dot(mask_a * (b_u-u_c) + mask_b * (b_l-l_c), kernel_neg_gamma) +\
                           K.dot(l_c, kernel_pos)+ K.dot(mask_a*u_c + mask_b*l_c, kernel_neg)
                    """
                    b_u_ = K.dot(self.alpha_u_f * b_u + (1 - self.alpha_u_f) * u_c, kernel_pos) + K.dot(
                        mask_a * (self.alpha_l_f * b_l + (1 - self.alpha_l_f) * l_c)
                        + mask_b * (self.alpha_u_f * b_u + (1 - self.alpha_u_f) * u_c),
                        kernel_neg,
                    )

                    b_l_ = K.dot(self.alpha_l_f * b_l + (1 - self.alpha_l_f) * l_c, kernel_pos) + K.dot(
                        mask_a * (self.alpha_u_f * b_u + (1 - self.alpha_u_f) * u_c)
                        + mask_b * (self.alpha_l_f * b_l + (1 - self.alpha_l_f) * l_c),
                        kernel_neg,
                    )
                    """
                else:
                    # if not self.n_subgrad or self.mode == F_FORWARD.name:
                    b_u_ = K.dot(b_u, kernel_pos) + K.dot(mask_a * b_l + mask_b * b_u, kernel_neg)
                    b_l_ = K.dot(b_l, kernel_pos) + K.dot(mask_a * b_u + mask_b * b_l, kernel_neg)

                # else:
                if self.n_subgrad and self.mode == F_HYBRID.name:

                    b_u_0_a = K.expand_dims(b_u, -1) * kernel_pos_
                    # b_u_0_ = K.sum(b_u_0_a, 1)
                    b_u_1_a = K.expand_dims(b_l, -1) * kernel_neg_
                    # b_u_1_ = K.sum(b_u_1_a, 1)
                    # b_u_ = b_u_0_ + b_u_1_

                    b_l_0_a = K.expand_dims(b_l, -1) * kernel_pos_
                    # b_l_0_ = K.sum(b_l_0_a, 1)
                    b_l_1_a = K.expand_dims(b_u, -1) * kernel_neg_
                    # b_l_1_ = K.sum(b_l_1_a, 1)
                    # b_l_ = b_l_0_ + b_l_1_

                mask_a = K.expand_dims(mask_a, 1)
                mask_b = K.expand_dims(mask_b, 1)

                # if not self.n_subgrad or self.mode == F_FORWARD.name:
                if self.finetune and self.mode == F_HYBRID.name:

                    w_u_ = K.dot(w_u, kernel_pos_alpha) + K.dot(mask_a * w_l + mask_b * w_u, kernel_neg_alpha)
                    w_l_ = K.dot(w_l, kernel_pos_gamma) + K.dot(mask_a * w_u + mask_b * w_l, kernel_neg_gamma)


                    """
                    w_u_ = K.dot(self.alpha_u_f * w_u, kernel_pos) + K.dot(
                        mask_a * self.alpha_l_f * w_l + mask_b * self.alpha_u_f * w_u, kernel_neg
                    )
                    w_l_ = K.dot(self.alpha_l_f * w_l, kernel_pos) + K.dot(
                        mask_a * self.alpha_u_f * w_u + mask_b * self.alpha_l_f * w_l, kernel_neg
                    )
                    """
                else:
                    w_u_ = K.dot(w_u, kernel_pos) + K.dot(mask_a * w_l + mask_b * w_u, kernel_neg)
                    w_l_ = K.dot(w_l, kernel_pos) + K.dot(mask_a * w_u + mask_b * w_l, kernel_neg)


                # else:
                """
                if self.n_subgrad and self.mode == F_HYBRID.name:
                    kernel_pos_ = K.expand_dims(kernel_pos_, 1)
                    kernel_neg_ = K.expand_dims(kernel_neg_, 1)

                    w_u_0_a = K.expand_dims(w_u, -1) * kernel_pos_
                    # w_u_0_ = K.sum(w_u_0_a, 2)
                    # w_u_1_a = K.expand_dims(mask_a * w_l + mask_b * w_u, -1) * kernel_neg_
                    w_u_1_a = K.expand_dims(w_l, -1) * kernel_neg_
                    # w_u_1_ = K.sum(w_u_1_a, 2)
                    # w_u_ = w_u_0_ + w_u_1_

                    w_l_0_a = K.expand_dims(w_l, -1) * kernel_pos_
                    # w_l_0_ = K.sum(w_l_0_a, 2)
                    # w_l_1_a = K.expand_dims(mask_a * w_u + mask_b * w_l, -1) * kernel_neg_
                    w_l_1_a = K.expand_dims(w_u, -1) * kernel_neg_
                    # w_l_1_ = K.sum(w_l_1_a, 2)
                    # w_l_ = w_l_0_ + w_l_1_

                    # if not self.fast and self.n_subgrad and self.mode == F_HYBRID.name:
                    # test whether the layer is linear:
                    # if it is: no need to use subgrad
                    convex_l_0 = (l_c_0_a, w_l_0_a, b_l_0_a)  # ???
                    convex_l_1 = (l_c_1_a, w_l_1_a, b_l_1_a)
                    convex_u_0 = (-u_c_0_a, -w_u_0_a, -b_u_0_a)
                    convex_u_1 = (-u_c_1_a, -w_u_1_a, -b_u_1_a)

                    # import pdb; pdb.set_trace()
                    l_sub = grad_descent(x_0, convex_l_0, convex_l_1, self.convex_domain, n_iter=self.n_subgrad)
                    u_sub = grad_descent(x_0, convex_u_0, convex_u_1, self.convex_domain, n_iter=self.n_subgrad)
                    u_c_ = u_sub
                    # u_c_ = K.minimum(u_c_, u_sub)
                    # l_c_ = K.maximum(l_c_, l_sub)
                """

        if self.use_bias:
            if self.mode in [F_HYBRID.name, F_IBP.name]:
                u_c_ = K.bias_add(u_c_, self.bias, data_format="channels_last")
                l_c_ = K.bias_add(l_c_, self.bias, data_format="channels_last")
            if self.mode in [F_FORWARD.name, F_HYBRID.name]:
                b_u_ = K.bias_add(b_u_, self.bias, data_format="channels_last")
                b_l_ = K.bias_add(b_l_, self.bias, data_format="channels_last")

            y_ = K.bias_add(y_, self.bias, data_format="channels_last")

            if self.dc_decomp:
                h_ = K.bias_add(h_, K.maximum(0.0, self.bias), data_format="channels_last")
                g_ = K.bias_add(g_, K.minimum(0.0, self.bias), data_format="channels_last")

        if self.mode == F_HYBRID.name and self.n_subgrad == 0:
            upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
            lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)

            l_c_ = K.maximum(lower_, l_c_)
            u_c_ = K.minimum(upper_, u_c_)

        if self.mode == F_HYBRID.name:
            output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == F_FORWARD.name:
            output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
        if self.mode == F_IBP.name:
            output = [y_, x_0, u_c_, l_c_]

        if self.dc_decomp:
            output += [h_, g_]

        return output

    def call(self, inputs):
        """
        :param inputs: list of tensors
        :return:
        """

        output = self.call_linear(inputs)

        if self.activation is not None and self.activation_name!='linear':

            if self.finetune:
                if self.activation_name[:4]!='relu':

                    output = self.activation(
                        output,
                        dc_decomp=self.dc_decomp,
                        convex_domain=self.convex_domain,
                        mode=self.mode,
                        finetune=[self.beta_u_f_, self.beta_l_f_]
                    )
                else:
                    output = self.activation(
                        output,
                        dc_decomp=self.dc_decomp,
                        convex_domain=self.convex_domain,
                        mode=self.mode,
                        finetune=self.beta_l_f_
                    )
            else:
                output = self.activation(
                    output,
                    dc_decomp=self.dc_decomp,
                    convex_domain=self.convex_domain,
                    mode=self.mode,
                )

        return output

    def compute_output_shape(self, input_shape):
        """
        :param input_shape:
        :return:
        """

        assert len(input_shape) == self.nb_tensors

        for i in range(self.nb_tensors):
            assert input_shape[i] and len(input_shape[i]) >= 2
            assert input_shape[i][-1]

        output_shape = [list(elem) for elem in input_shape[: self.nb_tensors]]
        for i, elem in enumerate(output_shape):
            if i == 1:
                # the convex domain is unchanged
                continue
            elem[-1] = self.units
        output_shape = [tuple(elem) for elem in output_shape]

        return output_shape

    def reset_layer(self, dense):
        """
        :param dense:
        :return:
        """
        # assert than we have the same configuration
        assert isinstance(dense, Dense), "wrong type of layers..."
        if dense.built:

            params = dense.get_weights()
            if self.finetune:
                params += self.get_weights()[2:]
            self.set_weights(params)
        else:
            raise ValueError("the layer {} has not been built yet".format(dense.name))

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
                if self.activation_name=='linear':
                    self._trainable_weights = self._trainable_weights[:-2]
                else:
                    if self.activation_name[:4]=='relu':
                        self._trainable_weights = self._trainable_weights[:-3]
                    else:
                        self._trainable_weights = self._trainable_weights[:-4]
                self.frozen_alpha = True

    def unfreeze_alpha(self):
        if not self.frozen_alpha:
            if self.finetune and self.mode == F_HYBRID.name:
                self._trainable_weights += [self.alpha_, self.gamma_]
                if self.activation_name!='linear':
                    if self.activation_name[:4]=='relu':
                        self._trainable_weights += [self.beta_u_f_, self.beta_l_f_]
                    else:
                        self._trainable_weights += [self.beta_u_f_, self.beta_l_f_]
            self.frozen_alpha = False

    def reset_finetuning(self):
        if self.finetune and self.mode == F_HYBRID.name:

            K.set_value(self.alpha_, np.ones_like(self.alpha_.value()))
            K.set_value(self.gamma_, np.ones_like(self.gamma_.value()))

            if self.activation_name!='linear':
                if self.activation_name[:4]=='relu':
                    K.set_value(self.beta_l_f_, np.ones_like(self.beta_l_f_.value()))
                else:
                    K.set_value(self.beta_u_f_, np.ones_like(self.beta_u_f_.value()))
                    K.set_value(self.beta_l_f_, np.ones_like(self.beta_l_f_.value()))


class DecomonActivation(Activation, DecomonLayer):
    """
    Forward LiRPA implementation of Activation layers.
    See Keras official documentation for further details on the Activation operator
    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):

        super(DecomonActivation, self).__init__(mode=mode, **kwargs)
        activation = kwargs["activation"]
        self.supports_masking = True
        self.activation = activations.get(activation)
        self.activation_name = activation

    def build(self, input_shape):

        if self.finetune:
            n_dim = len(input_shape[0])-1
            shape = [input_shape[0][i] for i in range(1, n_dim)]

            if self.activation_name != 'linear':

                if self.activation_name[:4] != 'relu':
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

    def call(self, input):

        if self.finetune:
            if self.activation_name[:4] != 'relu':
                return self.activation(input, mode=self.mode, finetune=self.beta_l_f_)
            else:
                return self.activation(input, mode=self.mode, finetune=[self.beta_u_f, self.beta_l_f_])
        else:
            return self.activation(input, mode=self.mode)


class DecomonFlatten(Flatten, DecomonLayer):
    """
    Forward LiRPA implementation of Flatten layers.
    See Keras official documentation for further details on the Flatten operator
    """

    def __init__(self, data_format=None, mode=F_HYBRID.name, **kwargs):
        """

        :param data_format:
        :param kwargs:

        """
        super(DecomonFlatten, self).__init__(data_format=data_format, mode=mode, **kwargs)

        if self.mode == F_HYBRID.name:
            self.input_spec = [
                InputSpec(min_ndim=1),  # y
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u
                InputSpec(min_ndim=2),  # w_u
                InputSpec(min_ndim=2),  # b_u
                InputSpec(min_ndim=1),  # l
                InputSpec(min_ndim=2),  # w_l
                InputSpec(min_ndim=2),  # b_l
            ]
        elif self.mode == F_IBP.name:
            self.input_spec = [
                InputSpec(min_ndim=1),  # y
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u
                InputSpec(min_ndim=1),  # l
            ]
        elif self.mode == F_FORWARD.name:
            self.input_spec = [
                InputSpec(min_ndim=1),  # y
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=2),  # w_u
                InputSpec(min_ndim=2),  # b_u
                InputSpec(min_ndim=2),  # w_l
                InputSpec(min_ndim=2),  # b_l
            ]
        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=1), InputSpec(min_ndim=1)]

    def build(self, input_shape):
        """

        :param self:
        :param input_shape:
        :return:
        """
        return None

    def call(self, inputs):

        op = super(DecomonFlatten, self).call

        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = op(h)
            g_ = op(g)
        if self.mode == F_HYBRID.name:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs
        elif self.mode == F_IBP.name:
            y, x_0, u_c, l_c = inputs
        elif self.mode == F_FORWARD.name:
            y, x_0, w_u, b_u, w_l, b_l = inputs

        y_ = op(y)
        if self.mode in [F_HYBRID.name, F_IBP.name]:
            u_c_ = op(u_c)
            l_c_ = op(l_c)

        if self.mode in [F_HYBRID.name, F_FORWARD.name]:
            b_u_ = op(b_u)
            b_l_ = op(b_l)

            output_shape = np.prod(list(K.int_shape(y_))[1:])
            input_dim = K.int_shape(x_0)[-1]

            w_u_ = K.reshape(w_u, (-1, input_dim, output_shape))
            w_l_ = K.reshape(w_l, (-1, input_dim, output_shape))

        if self.mode == F_HYBRID.name:
            output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == F_IBP.name:
            output = [y_, x_0, u_c_, l_c_]
        if self.mode == F_FORWARD.name:
            output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

        if self.dc_decomp:
            output += [h_, g_]

        return output


class DecomonBatchNormalization(BatchNormalization, DecomonLayer):
    """
    Forward LiRPA implementation of Batch Normalization layers.
    See Keras official documentation for further details on the BatchNormalization operator
    """

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        mode=F_HYBRID.name,
        **kwargs,
    ):
        super(DecomonBatchNormalization, self).__init__(
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
            mode=mode,
            **kwargs,
        )

    def build(self, input_shape):
        super(DecomonBatchNormalization, self).build(input_shape[0])

        self.input_spec = [InputSpec(min_ndim=len(elem)) for elem in input_shape]

    def compute_output_shape(self, input_shape):

        output_shape_ = super(DecomonBatchNormalization, self).compute_output_shape(input_shape[0])
        x_shape = input_shape[1]
        input_dim = x_shape[-1]

        output = []
        if self.mode == F_IBP.name:
            output = [output_shape_, x_shape, output_shape_, output_shape_]
        if self.mode in [F_HYBRID.name, F_FORWARD.name]:
            w_shape = list(output_shape_)[:, None]
            w_shape[:, 0] = input_dim
            if self.mode == F_FORWARD.name:
                output = [output_shape_, x_shape, w_shape, output_shape_, w_shape, output_shape_]
            else:
                output = [
                    output_shape_,
                    x_shape,
                    output_shape_,
                    w_shape,
                    output_shape_,
                    output_shape_,
                    w_shape,
                    output_shape_,
                ]

        if self.dc_decomp:
            output += [output_shape_, output_shape_]
        return output

    def call(self, inputs, training=None):

        if training is None:
            training = K.learning_phase()

        if training:
            raise NotImplementedError("not working during training")

        call_op = super(DecomonBatchNormalization, self).call

        if self.dc_decomp:
            raise NotImplementedError()
            h, g = inputs[-2:]

        if self.mode == F_HYBRID.name:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
        elif self.mode == F_IBP.name:
            y, x_0, u_c, l_c = inputs[:4]
        elif self.mode == F_FORWARD.name:
            y, x_0, w_u, b_u, w_l, b_l = inputs[:6]

        y_ = call_op(y, training=training)

        n_dim = len(y_.shape)
        tuple_ = [1] * n_dim
        for i, ax in enumerate(self.axis):
            tuple_[ax] = self.moving_mean.shape[i]

        gamma_ = K.reshape(self.gamma + 0.0, tuple_)
        beta_ = K.reshape(self.beta + 0.0, tuple_)
        moving_mean_ = K.reshape(self.moving_mean + 0.0, tuple_)
        moving_variance_ = K.reshape(self.moving_variance + 0.0, tuple_)

        if self.mode in [F_HYBRID.name, F_IBP.name]:

            u_c_0 = (u_c - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)  # + beta_
            l_c_0 = (l_c - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)

            u_c_ = K.maximum(0.0, gamma_) * u_c_0 + K.minimum(0.0, gamma_) * l_c_0 + beta_
            l_c_ = K.maximum(0.0, gamma_) * l_c_0 + K.minimum(0.0, gamma_) * u_c_0 + beta_

        if self.mode in [F_HYBRID.name, F_FORWARD.name]:

            b_u_0 = (b_u - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)
            b_l_0 = (b_l - moving_mean_) / K.sqrt(moving_variance_ + self.epsilon)

            b_u_ = K.maximum(0.0, gamma_) * b_u_0 + K.minimum(0.0, gamma_) * b_l_0 + beta_
            b_l_ = K.maximum(0.0, gamma_) * b_l_0 + K.minimum(0.0, gamma_) * b_u_0 + beta_

            gamma_ = K.expand_dims(gamma_, 1)
            moving_variance_ = K.expand_dims(moving_variance_, 1)

            w_u_0 = w_u / K.sqrt(moving_variance_ + self.epsilon)
            w_l_0 = w_l / K.sqrt(moving_variance_ + self.epsilon)
            w_u_ = K.maximum(0.0, gamma_) * w_u_0 + K.minimum(0.0, gamma_) * w_l_0
            w_l_ = K.maximum(0.0, gamma_) * w_l_0 + K.minimum(0.0, gamma_) * w_u_0

        if self.mode == F_HYBRID.name:
            output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == F_IBP.name:
            output = [y_, x_0, u_c_, l_c_]
        if self.mode == F_FORWARD.name:
            output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

        return output

    def reset_layer(self, layer):
        """

        :param layer:
        :return:

        """
        assert isinstance(layer, BatchNormalization), "wrong type of layers..."
        params = layer.get_weights()
        self.set_weights(params)


class DecomonDropout(Dropout, DecomonLayer):
    """
    Forward LiRPA implementation of Dropout layers.
    See Keras official documentation for further details on the Dropout operator
    """

    def __init__(self, rate, noise_shape=None, seed=None, mode=F_HYBRID.name, **kwargs):
        super(DecomonDropout, self).__init__(rate=rate, noise_shape=noise_shape, seed=seed, mode=mode, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        super(DecomonDropout, self).build(input_shape[0])
        self.input_spec = [InputSpec(min_ndim=len(elem)) for elem in input_shape]

    def call(self, inputs, training=None):

        if training is None:
            training = K.learning_phase()

        if training:

            raise NotImplementedError("not working during training")

        return inputs

        call_op = super(DecomonDropout, self).call

        if self.mode == F_HYBRID.name:
            if self.dc_decomp:
                y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]
        elif self.mode == F_IBP.name:
            if self.dc_decomp:
                y, x_0, u_c, l_c, h, g = inputs[: self.nb_tensors]
            else:
                y, x_0, u_c, l_c = inputs[: self.nb_tensors]
        elif self.mode == F_FORWARD.name:
            if self.dc_decomp:
                y, x_0, w_u, b_u, w_l, b_l, h, g = inputs[: self.nb_tensors]
            else:
                y, x_0, w_u, b_u, w_l, b_l = inputs[: self.nb_tensors]

        y_ = call_op(y, training=training)
        if self.mode in [F_HYBRID.name, F_IBP.name]:
            u_c_ = call_op(u_c, training=training)
            l_c_ = call_op(l_c, training=training)

        if self.mode in [F_HYBRID.name, F_FORWARD.name]:
            b_u_ = call_op(b_u, training=training)
            b_l_ = call_op(b_l, training=training)
            input_dim = w_u.shape[1]
            w_u_list = tf.split(w_u, input_dim, 1)
            w_l_list = tf.split(w_l, input_dim, 1)
            w_u_ = K.concatenate([call_op(w_u_i, training=training) for w_u_i in w_u_list], 1)
            w_l_ = K.concatenate([call_op(w_l_i, training=training) for w_l_i in w_l_list], 1)

        if self.mode == F_HYBRID.name:
            output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == F_IBP.name:
            output = [y_, x_0, u_c_, l_c_]
        if self.mode == F_FORWARD.name:
            output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]

        return output


def to_monotonic(
    layer,
    input_dim,
    dc_decomp=False,
    convex_domain={},
    finetune=False,
    IBP=True,
    forward=True,
):
    """Transform a standard keras layer into a Decomon layer.

    Type of layer is tested to know how to transform it into a MonotonicLayer of the good type.
    If type is not treated yet, raises an TypeError

    :param layer: a Keras Layer
    :param input_dim: either an integer or a tuple that represents the dim of the input convex domain
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param IBP: boolean that indicates whether we propagate constant bounds
    :param forward: boolean that indicates whether we propagate affine bounds
    :return: the associated DecomonLayer
    :raises: TypeError
    """

    # get class name
    class_name = layer.__class__.__name__

    # check if layer has a built argument that built is set to True
    if hasattr(layer, "built"):
        if not layer.built:
            raise ValueError("the layer {} has not been built yet".format(layer.name))

    if isinstance(layer, Merge):
        return to_monotonic_merge(layer, input_dim, dc_decomp, convex_domain, finetune, IBP, forward)

    # do case by case for optimizing

    for k in range(3):
        # two runs before sending a failure
        if k == 2:
            # the immediate parent is not a native Keras class
            raise KeyError("unknown class {}".format(class_name))
        try:
            monotonic_class_name = "Decomon{}".format(class_name)
            config_layer = layer.get_config()
            config_layer["name"] = layer.name + "_monotonic"
            config_layer["dc_decomp"] = dc_decomp
            config_layer["convex_domain"] = convex_domain

            mode = F_HYBRID.name
            if IBP and not forward:
                mode = F_IBP.name
            if not IBP and forward:
                mode = F_FORWARD.name

            config_layer["mode"] = mode
            config_layer["finetune"] = finetune
            """
            if k==1:
                attr_parent = inspect.getargspec(layer.__class__.__bases__[0].__dict__['__init__'])[0]
                attr_child = inspect.getargspec(layer.__class__.__dict__['__init__'])[0]

                specific_item = [elem for elem in attr_child if not elem in attr_parent]
                import pdb; pdb.set_trace()
            """
            layer_monotonic = globals()[monotonic_class_name].from_config(config_layer)
            break
        except KeyError:
            """
            # retrieve first parent to apply LiRPA decomposition
            class_name_= class_name
            class_name = layer.__class__.__bases__[0].__name__
            warnings.warn('unknown class {} as a native Keras class. We replace it by its direct parent class {}'.format(class_name_, class_name))
            """
            if hasattr(layer, "vanilla_export"):
                layer_ = layer.vanilla_export()
                layer_(layer.input)
                layer = layer_
                class_name = layer.__class__.__name__

    input_shape = list(layer.input_shape)[1:]
    if isinstance(input_dim, tuple):
        x_shape = Input(input_dim)
        input_dim = input_dim[-1]
    else:
        x_shape = Input((input_dim,))

    w_shape = Input(tuple([input_dim] + input_shape))
    y_shape = Input(tuple(input_shape))

    if mode == F_HYBRID.name:
        input_ = [y_shape, x_shape, y_shape, w_shape, y_shape, y_shape, w_shape, y_shape]
    elif mode == F_IBP.name:
        input_ = [y_shape, x_shape, y_shape, y_shape]
    elif mode == F_FORWARD.name:
        input_ = [y_shape, x_shape, w_shape, y_shape, w_shape, y_shape]

    if dc_decomp:
        input_ += [y_shape, y_shape]

    layer_monotonic(input_)
    layer_monotonic.reset_layer(layer)

    return layer_monotonic


# Aliases
MonotonicConvolution2D = DecomonConv2D
