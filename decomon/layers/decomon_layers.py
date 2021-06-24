from __future__ import absolute_import
import tensorflow as tf
from .core import DecomonLayer
import tensorflow.keras.backend as K
from tensorflow.keras.backend import bias_add, conv2d
import numpy as np
from tensorflow.keras.constraints import NonNeg
from tensorflow.python.keras.engine.base_layer import InputSpec
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
)
from decomon.layers import activations
from decomon.layers.utils import NonPos, MultipleConstraint, Project_initializer_pos, Project_initializer_neg
from tensorflow.python.keras.utils import conv_utils
from decomon.layers.utils import get_upper, get_lower
from .maxpooling import DecomonMaxPooling2D
from .utils import grad_descent
from .core import F_FORWARD, F_IBP, F_HYBRID


class DecomonConv2D(Conv2D, DecomonLayer):
    def __init__(self, filters, kernel_size, mode=F_HYBRID.name, **kwargs):
        """

          :param filters: Integer, the dimensionality of the output space (i.e. the number of
        output filters in the convolution).
          :param kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to specify
        the same value for all spatial dimensions.
          :param kwargs: refer to the official Keras documentation for further information
        """
        activation = kwargs["activation"]
        if "activation" in kwargs:
            kwargs["activation"] = None
        super(DecomonConv2D, self).__init__(filters=filters, kernel_size=kernel_size, mode=mode, **kwargs)
        self.kernel_constraint_pos_ = NonNeg()
        self.kernel_constraint_neg_ = NonPos()

        if self.grad_bounds:
            raise NotImplementedError()
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
        if self.grad_bounds:
            raise NotImplementedError()

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
        :param inputs: list of at least 8 to 10 tensors [h, g, x_min, x_max, u_c, W_u, b_u, l_c, W_l, b_l]
        :param kwargs:
        :return: List of at least 8 tensors
        """

        if self.grad_bounds:
            raise NotImplementedError()

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

            else:
                # check for linearity
                mask_b = 0 * y
                if self.mode in [F_HYBRID.name]:  # F_FORWARD.name
                    x_max = get_upper(x_0, w_u - w_l, b_u - b_l, self.convex_domain)
                    mask_b = 1.0 - K.sign(x_max)
                mask_a = 1.0 - mask_b

                b_u_ = conv_pos(b_u) + conv_neg(mask_a * b_l + mask_b * b_u)
                b_l_ = conv_pos(b_l) + conv_neg(mask_a * b_u + mask_b * b_l)

                mask_a = K.expand_dims(mask_a, 1)
                mask_b = K.expand_dims(mask_b, 1)

                def step_pos(x, _):
                    return conv_pos(x), []

                def step_neg(x, _):
                    return conv_neg(x), []

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

        :param inputs: list of 8 to 10 tensors [h, g, x_min, x_max, u_c, W_u, b_u, l_c, W_l, b_l]
        :return: List of at least 8 tensors

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

        b_u_shape_, b_l_shape_, u_c_shape, l_c_shape = [output_shape] * 4
        input_dim = x_0_shape[-1]
        w_u_shape_, w_l_shape_ = [tuple([output_shape[0], input_dim] + list(output_shape)[1:])] * 2

        output_shape_ = [
            y_shape,
            x_0_shape,
            u_c_shape,
            w_u_shape_,
            b_u_shape_,
            l_c_shape,
            w_l_shape_,
            b_l_shape_,
        ]

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
        self.set_weights(layer.get_weights())


class DecomonDense(Dense, DecomonLayer):
    """Just your regular densely-connected NN layer
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Arguments
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix
    (see [initializers](../initializers.md)).
    bias_initializer: Initializer for the bias vector
    (see [initializers](../initializers.md)).
    kernel_regularizer: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
    bias_regularizer: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
    activity_regularizer: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
    kernel_constraint: Constraint function applied to
    the `kernel` weights matrix
    (see [constraints](../constraints.md)).
    bias_constraint: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).
    # Input shape
    nD tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
    nD tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
    see the official Keras document for complementary information
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

    def build(self, input_shape):
        """
        :param input_shape: list of input shape
        :return:
        """
        if self.grad_bounds:
            raise NotImplementedError()

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
        if self.grad_bounds:
            raise NotImplementedError()

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called " "on a list of inputs.")

        kernel_pos = K.maximum(0.0, self.kernel)
        kernel_neg = K.minimum(0.0, self.kernel)

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

        if self.n_subgrad and self.mode == F_HYBRID.name:
            kernel_pos_ = K.expand_dims(kernel_pos, 0)
            kernel_neg_ = K.expand_dims(kernel_neg, 0)

        if self.mode in [F_HYBRID.name, F_IBP.name]:

            if not self.n_subgrad or self.mode == F_IBP.name:

                u_c_ = K.dot(u_c, kernel_pos) + K.dot(l_c, kernel_neg)
                l_c_ = K.dot(l_c, kernel_pos) + K.dot(u_c, kernel_neg)
            else:

                u_c_0_a = K.expand_dims(u_c, -1) * kernel_pos_
                u_c_0_ = K.sum(u_c_0_a, 1)  # K.dot(u_c, self.kernel_pos)
                u_c_1_a = K.expand_dims(l_c, -1) * kernel_neg_
                u_c_1_ = K.sum(u_c_1_a, 1)
                u_c_ = u_c_0_ + u_c_1_

                l_c_0_a = K.expand_dims(l_c, -1) * kernel_pos_
                l_c_0_ = K.sum(l_c_0_a, 1)
                l_c_1_a = K.expand_dims(u_c, -1) * kernel_neg_
                l_c_1_ = K.sum(l_c_1_a, 1)
                l_c_ = l_c_0_ + l_c_1_

        #######
        if self.mode in [F_HYBRID.name, F_FORWARD.name]:

            if len(w_u.shape) != len(b_u.shape):
                # first layer, it is necessary linear

                if not self.n_subgrad or self.mode == F_FORWARD.name:
                    b_u_ = K.dot(b_u, kernel_pos) + K.dot(mask_a * b_l + mask_b * b_u, kernel_neg)
                    b_l_ = K.dot(b_l, kernel_pos) + K.dot(mask_a * b_u + mask_b * b_l, kernel_neg)
                else:

                    b_u_0_a = K.expand_dims(b_u, -1) * kernel_pos_
                    b_u_0_ = K.sum(b_u_0_a, 1)
                    b_u_1_a = K.expand_dims(mask_a * b_l + mask_b * b_u, -1) * kernel_neg_
                    b_u_1_ = K.sum(b_u_1_a, 1)
                    b_u_ = b_u_0_ + b_u_1_

                    b_l_0_a = K.expand_dims(mask_a * b_l + mask_b * b_l, -1) * kernel_pos_
                    b_l_0_ = K.sum(b_l_0_a, 1)
                    b_l_1_a = K.expand_dims(mask_a * b_u + mask_b * b_l, -1) * kernel_neg_
                    b_l_1_ = K.sum(b_l_1_a, 1)
                    b_l_ = b_l_0_ + b_l_1_

                mask_a = K.expand_dims(mask_a, 1)
                mask_b = K.expand_dims(mask_b, 1)

                if not self.n_subgrad or self.mode == F_FORWARD.name:
                    w_u_ = K.dot(w_u, kernel_pos) + K.dot(mask_a * w_l + mask_b * w_u, kernel_neg)
                    w_l_ = K.dot(w_l, kernel_pos) + K.dot(mask_a * w_u + mask_b * w_l, kernel_neg)
                else:
                    kernel_pos_ = K.expand_dims(kernel_pos_, 1)
                    kernel_neg_ = K.expand_dims(kernel_neg_, 1)

                    w_u_0_a = K.expand_dims(w_u, -1) * kernel_pos_
                    w_u_0_ = K.sum(w_u_0_a, 2)
                    w_u_1_a = K.expand_dims(mask_a * w_l + mask_b * w_u, -1) * kernel_neg_
                    w_u_1_ = K.sum(w_u_1_a, 2)
                    w_u_ = w_u_0_ + w_u_1_

                    w_l_0_a = K.expand_dims(w_l, -1) * kernel_pos_
                    w_l_0_ = K.sum(w_l_0_a, 2)
                    w_l_1_a = K.expand_dims(mask_a * w_u + mask_b * w_l, -1) * kernel_neg_
                    w_l_1_ = K.sum(w_l_1_a, 2)
                    w_l_ = w_l_0_ + w_l_1_

                    # if not self.fast and self.n_subgrad and self.mode == F_HYBRID.name:
                    # test whether the layer is linear:
                    # if it is: no need to use subgrad
                    convex_l_0 = (l_c_0_a, w_l_0_a, b_l_0_a)  # ???
                    convex_l_1 = (l_c_1_a, w_l_1_a, b_l_1_a)
                    convex_u_0 = (-u_c_0_a, -w_u_0_a, -b_u_0_a)
                    convex_u_1 = (-u_c_1_a, -w_u_1_a, -b_u_1_a)
                    # import pdb; pdb.set_trace()
                    l_sub = grad_descent(x_0, convex_l_0, convex_l_1, self.convex_domain, n_iter=self.n_subgrad)
                    u_sub = -grad_descent(x_0, convex_u_0, convex_u_1, self.convex_domain, n_iter=self.n_subgrad)

                    u_c_ = K.minimum(u_c_, u_sub)
                    l_c_ = K.maximum(l_c_, l_sub)

            ########
            else:
                w_u_ = K.expand_dims(0 * (y_), 1) + K.expand_dims(self.kernel, 0)
                w_l_ = w_u_
                b_u_ = 0 * y_
                b_l_ = b_u_

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

        if self.mode == F_HYBRID.name:
            upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
            u_c_ = K.minimum(upper_, u_c_)

            lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)
            l_c_ = K.maximum(lower_, l_c_)

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

        if self.activation is not None:
            output = self.activation(
                output,
                dc_decomp=self.dc_decomp,
                grad_bounds=self.grad_bounds,
                convex_domain=self.convex_domain,
                mode=self.mode,
                fast=self.fast,
            )

        return output

    def compute_output_shape(self, input_shape):
        """
        :param input_shape:
        :return:
        """

        if self.grad_bounds:
            raise NotImplementedError()

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
            self.set_weights(params)
        else:
            raise ValueError("the layer {} has not been built yet".format(dense.name))


class DecomonActivation(Activation, DecomonLayer):
    """Applies an activation function to an output.
    # Arguments
    activation: name of activation function to use
    (see: [activations](../activations.md)),
    or alternatively, a Theano or TensorFlow operation.
    # Input shape
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
    # Output shape
    Same shape as input.

    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):
        """

        :param kwargs:

        """
        super(DecomonActivation, self).__init__(mode=mode, **kwargs)
        activation = kwargs["activation"]
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, input):
        return self.activation(input, mode=self.mode)


class DecomonFlatten(Flatten, DecomonLayer):
    def __init__(self, data_format=None, mode=F_HYBRID.name, **kwargs):
        """

        :param data_format:
        :param kwargs:

        """
        super(DecomonFlatten, self).__init__(data_format=data_format, mode=mode, **kwargs)

        if self.grad_bounds:
            raise NotImplementedError()

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

        if self.grad_bounds:
            raise NotImplementedError()

    def call(self, inputs):

        if self.grad_bounds:
            raise NotImplementedError()

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


class DecomonReshape(Reshape, DecomonLayer):
    def __init__(self, target_shape, mode=F_HYBRID.name, **kwargs):
        """

        :param data_format:
        :param kwargs:

        """
        super(DecomonReshape, self).__init__(target_shape=target_shape, mode=mode, **kwargs)
        if self.grad_bounds:
            raise NotImplementedError()

        if self.mode == F_HYBRID.name:
            self.input_spec = [
                InputSpec(min_ndim=1),  # y
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # l_c
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]
        elif self.mode == F_IBP.name:
            self.input_spec = [
                InputSpec(min_ndim=1),  # y
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # l_c
            ]
        if self.mode == F_FORWARD.name:
            self.input_spec = [
                InputSpec(min_ndim=1),  # y
                InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # w_u
                InputSpec(min_ndim=1),  # b_u
                InputSpec(min_ndim=1),  # w_l
                InputSpec(min_ndim=1),  # b_l
            ]

        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=1), InputSpec(min_ndim=1)]

    def build(self, input_shape):
        """

        :param self:
        :param input_shape:
        :return:
        """

        if self.grad_bounds:
            raise NotImplementedError()

        y_input_shape = input_shape[0]
        super(DecomonReshape, self).build(y_input_shape)

    def call(self, inputs):

        if self.grad_bounds:
            raise NotImplementedError()

        op = super(DecomonReshape, self).call

        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = op(h)
            g_ = op(g)

        if self.mode == F_HYBRID.name:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
        if self.mode == F_IBP.name:
            y, x_0, u_c, l_c = inputs[:4]
        if self.mode == F_FORWARD.name:
            y, x_0, w_u, b_u, w_l, b_l = inputs[:6]

        y_ = op(y)
        if self.mode in [F_IBP.name, F_HYBRID.name]:
            u_c_ = op(u_c)
            l_c_ = op(l_c)

        if self.mode in [F_HYBRID.name, F_FORWARD.name]:
            b_u_ = op(b_u)
            b_l_ = op(b_l)

            if len(w_u.shape) == len(b_u.shape):
                w_u_ = op(w_u)
                w_l_ = op(w_l)
            else:

                def step_func(x, _):
                    return op(x), _

                w_u_ = K.rnn(step_function=step_func, inputs=w_u, initial_states=[], unroll=False)[1]
                w_l_ = K.rnn(step_function=step_func, inputs=w_l, initial_states=[], unroll=False)[1]

        if self.mode == F_HYBRID.name:
            output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == F_FORWARD.name:
            output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
        if self.mode == F_IBP.name:
            output = [y_, x_0, u_c_, l_c_]

        if self.dc_decomp:
            output += [h_, g_]

        return output


class DecomonBatchNormalization(BatchNormalization, DecomonLayer):
    """Layer that normalizes its inputs.
    Batch normalization applies a transformation that maintains the mean output
    close to 0 and the output standard deviation close to 1.
    Importantly, batch normalization works differently during training and
    during inference.
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

        if self.grad_bounds:
            raise NotImplementedError()

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

        if self.grad_bounds:
            raise NotImplementedError()

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
    grad_bounds=False,
    n_subgrad=0,
    convex_domain={},
    IBP=True,
    forward=True,
    fast=True,
):
    """Transform a standard keras layer into a decomon layer.

    Type of layer is tested to know how to transform it into a MonotonicLayer of the good type.
    If type is not treated yet, raises an TypeError

    :param layer: a Keras Layer
    :param input_dim: either an integer or a tuple that represents the dim of the convex domain
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :return: the associated DecomonLayer
    :raises: TypeError

    """
    if grad_bounds:
        raise NotImplementedError()

    # get class name
    class_name = layer.__class__.__name__
    # check if layer has a built argument that built is set to True
    if hasattr(layer, "built"):
        if not layer.built:
            raise ValueError("the layer {} has not been built yet".format(layer.name))

    monotonic_class_name = "Decomon{}".format(class_name)
    config_layer = layer.get_config()
    config_layer["name"] = layer.name + "_monotonic"
    config_layer["dc_decomp"] = dc_decomp
    config_layer["grad_bounds"] = grad_bounds
    config_layer["convex_domain"] = convex_domain
    config_layer["n_subgrad"] = n_subgrad
    config_layer["fast"] = fast

    mode = F_HYBRID.name
    if IBP and not forward:
        mode = F_IBP.name
    if not IBP and forward:
        mode = F_FORWARD.name

    config_layer["mode"] = mode
    layer_monotonic = globals()[monotonic_class_name].from_config(config_layer)

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
