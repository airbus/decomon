from __future__ import absolute_import
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
)
from decomon.layers import activations
from decomon.layers.utils import NonPos, MultipleConstraint, Project_initializer_pos, Project_initializer_neg
from tensorflow.python.keras.utils import conv_utils
from decomon.layers.utils import time_distributed, get_upper, get_lower
from .maxpooling import DecomonMaxPooling2D


class DecomonConv2D(Conv2D, DecomonLayer):
    def __init__(self, filters, kernel_size, **kwargs):
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
        super(DecomonConv2D, self).__init__(filters=filters, kernel_size=kernel_size, **kwargs)
        self.kernel_constraint_pos_ = NonNeg()
        self.kernel_constraint_neg_ = NonPos()

        if self.grad_bounds:
            raise NotImplementedError()

        self.input_spec = [
            InputSpec(min_ndim=4),
            InputSpec(min_ndim=2),
            InputSpec(min_ndim=4),
            InputSpec(min_ndim=5),
            InputSpec(min_ndim=4),
            InputSpec(min_ndim=4),
            InputSpec(min_ndim=5),
            InputSpec(min_ndim=4),
        ]
        if self.dc_decomp:
            self.input_spec += [InputSpec(min_ndim=4), InputSpec(min_ndim=4)]

        self.kernel_pos = None
        self.kernel_neg = None
        self.kernel = None
        self.kernel_constraints_pos = None
        self.kernel_constraints_neg = None
        self.activation = activations.get(activation)
        self.bias = None

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

        self.kernel_constraints_pos = MultipleConstraint(self.kernel_constraint, self.kernel_constraint_pos_)
        self.kernel_constraints_neg = MultipleConstraint(self.kernel_constraint, self.kernel_constraint_neg_)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name="kernel_all",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        self.kernel_pos = self.add_weight(
            shape=kernel_shape,
            initializer=Project_initializer_pos(self.kernel_initializer),
            name="kernel_pos",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraints_pos,
        )
        self.kernel_neg = self.add_weight(
            shape=kernel_shape,
            initializer=Project_initializer_neg(self.kernel_initializer),
            name="kernel_neg",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraints_neg,
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

        # inputs tensors: h, g, x_min, x_max, W_u, b_u, W_l, b_l
        if channel_axis == -1:
            self.input_spec = [
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                InputSpec(min_ndim=2),  # x
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # u_c
                InputSpec(min_ndim=5, axes={channel_axis: input_dim}),  # w_u
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_u
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # l_c
                InputSpec(min_ndim=5, axes={channel_axis: input_dim}),  # w_l
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_l
            ]
            if self.dc_decomp:
                self.input_spec += [
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # h
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),
                ]  # g
        else:
            self.input_spec = [
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # y
                InputSpec(min_ndim=2),  # x
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # u_c
                InputSpec(min_ndim=5, axes={channel_axis + 1: input_dim}),  # w_u
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_u
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # l_c
                InputSpec(min_ndim=5, axes={channel_axis + 1: input_dim}),  # w_l
                InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # b_l
            ]
            if self.dc_decomp:
                self.input_spec += [
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),  # h
                    InputSpec(min_ndim=4, axes={channel_axis: input_dim}),
                ]  # g

        self.built = True

    def call(self, inputs, **kwargs):
        """

        :param inputs: list of at least 10 tensors [h, g, x_min, x_max, u_c, W_u, b_u, l_c, W_l, b_l]
        :return: List of at least 8 tensors

        """
        if self.grad_bounds:
            raise NotImplementedError()

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called " "on a list of inputs.")

        if self.dc_decomp:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs[: self.nb_tensors]
        else:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]

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
                self.kernel_pos,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        def conv_neg(x):
            return conv2d(
                x,
                self.kernel_neg,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
            )

        if self.dc_decomp:
            h_ = conv_pos(h) + conv_neg(g)
            g_ = conv_pos(g) + conv_neg(h)

        b_u_ = conv_pos(b_u) + conv_neg(b_l)

        b_l_ = conv_pos(b_l) + conv_neg(b_u)

        u_c_ = conv_pos(u_c) + conv_neg(l_c)

        l_c_ = conv_pos(l_c) + conv_neg(u_c)

        output_shape = b_u_.shape.as_list()[1:]

        w_u_ = time_distributed(w_u, conv_pos, output_shape) + time_distributed(w_l, conv_neg, output_shape)

        w_l_ = time_distributed(w_l, conv_pos, output_shape) + time_distributed(w_u, conv_neg, output_shape)

        # add bias
        if self.use_bias:
            b_u_ = bias_add(b_u_, self.bias, data_format=self.data_format)
            b_l_ = bias_add(b_l_, self.bias, data_format=self.data_format)
            u_c_ = bias_add(u_c_, self.bias, data_format=self.data_format)
            l_c_ = bias_add(l_c_, self.bias, data_format=self.data_format)
            y_ = bias_add(y_, self.bias, data_format=self.data_format)
            if self.dc_decomp:
                g_ = bias_add(g_, K.minimum(0.0, self.bias), data_format=self.data_format)
                h_ = bias_add(h_, K.maximum(0.0, self.bias), data_format=self.data_format)

        upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
        u_c_ = K.minimum(upper_, u_c_)

        lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)
        l_c_ = K.maximum(lower_, l_c_)

        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

        if self.dc_decomp:
            output += [h_, g_]

        # temporary fix until all activations are ready
        if self.activation is not None:
            output = self.activation(output, dc_decomp=self.dc_decomp)

        return output

    def compute_output_shape(self, input_shape):
        """

        :param input_shape:
        :return:

        """
        assert len(input_shape) == self.nb_tensors

        if self.dc_decomp:
            (
                y_shape,
                x_0_shape,
                u_c_shape,
                w_u_shape,
                b_u_shape,
                l_c_shape,
                w_l_shape,
                b_l_shape,
                h_shape,
                g_shape,
            ) = input_shape[: self.nb_tensors]
        else:
            (
                y_shape,
                x_0_shape,
                u_c_shape,
                w_u_shape,
                b_u_shape,
                l_c_shape,
                w_l_shape,
                b_l_shape,
            ) = input_shape[: self.nb_tensors]

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
        # assert min([elem_0 == elem_1 for (elem_0, elem_1) in
        #             zip(self.get_config(), layer.get_config())]), 'different configuration'

        params = layer.get_weights()
        # only 1 weight for conv2D without bias
        w = params[0]
        w_pos = np.maximum(0.0, params[0])
        w_neg = np.minimum(0.0, params[0])
        if self.use_bias:
            self.set_weights([w, w_pos, w_neg, params[1]])
        else:
            self.set_weights([w, w_pos, w_neg])


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

    def __init__(self, units, **kwargs):
        if "activation" not in kwargs:
            kwargs["activation"] = None
        activation = kwargs["activation"]
        kwargs["units"] = units
        super(DecomonDense, self).__init__(**kwargs)
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

        self.kernel_constraints_pos = MultipleConstraint(self.kernel_constraint, self.kernel_constraint_pos_)
        self.kernel_constraints_neg = MultipleConstraint(self.kernel_constraint, self.kernel_constraint_neg_)

        self.kernel_pos = self.add_weight(
            shape=(input_dim, self.units),
            initializer=Project_initializer_pos(self.kernel_initializer),
            name="kernel_pos",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraints_pos,
        )
        self.kernel_neg = self.add_weight(
            shape=(input_dim, self.units),
            initializer=Project_initializer_neg(self.kernel_initializer),
            name="kernel_neg",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraints_neg,
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

        # import pdb; pdb.set_trace()
        # 6 inputs tensors: h, g, h_min, h_max, g_min, g_max
        self.input_spec = [
            InputSpec(min_ndim=2, axes={-1: input_dim}),  # y
            InputSpec(min_ndim=2, axes={-1: input_x}),  # x_0
            InputSpec(min_ndim=2, axes={-1: input_dim}),  # u_c
            InputSpec(min_ndim=3, axes={-1: input_dim}),  # W_u
            InputSpec(min_ndim=2, axes={-1: input_dim}),  # b_u
            InputSpec(min_ndim=2, axes={-1: input_dim}),  # l_c
            InputSpec(min_ndim=3, axes={-1: input_dim}),  # W_l
            InputSpec(min_ndim=2, axes={-1: input_dim}),
        ]  # b_l

        if self.dc_decomp:
            self.input_spec += [
                InputSpec(min_ndim=2, axes={-1: input_dim}),  # h
                InputSpec(min_ndim=2, axes={-1: input_dim}),
            ]  # g

        self.built = True

    def call(self, inputs):
        """

        :param inputs: list of tensors
        :return:
        """

        if self.grad_bounds:
            raise NotImplementedError()

        if not isinstance(inputs, list):
            raise ValueError("A merge layer should be called " "on a list of inputs.")
        if self.dc_decomp:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs[: self.nb_tensors]
            h_ = K.dot(h, self.kernel_pos) + K.dot(g, self.kernel_neg)
            g_ = K.dot(g, self.kernel_pos) + K.dot(h, self.kernel_neg)
        else:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[: self.nb_tensors]

        y_ = K.dot(y, self.kernel_pos) + K.dot(y, self.kernel_neg)
        b_u_ = K.dot(b_u, self.kernel_pos) + K.dot(b_l, self.kernel_neg)
        b_l_ = K.dot(b_l, self.kernel_pos) + K.dot(b_u, self.kernel_neg)

        u_c_ = K.dot(u_c, self.kernel_pos) + K.dot(l_c, self.kernel_neg)
        l_c_ = K.dot(l_c, self.kernel_pos) + K.dot(u_c, self.kernel_neg)

        w_u_ = K.dot(w_u, self.kernel_pos) + K.dot(w_l, self.kernel_neg)
        w_l_ = K.dot(w_l, self.kernel_pos) + K.dot(w_u, self.kernel_neg)

        # gradient descent: optional
        # w_u_tensor = (
        #    K.expand_dims(w_u, -1) * self.kernel_pos[None, None]
        #    + K.expand_dims(w_l, -1) * self.kernel_neg[None, None]
        # )
        # b_u_tensor = (
        #    K.expand_dims(b_u, -1) * self.kernel_pos[None]
        #    + K.expand_dims(b_l, -1) * self.kernel_neg[None]
        # )
        # u_tensor = (
        #    K.expand_dims(u_c, -1) * self.kernel_pos[None]
        #    + K.expand_dims(l_c, -1) * self.kernel_neg[None]
        # )
        # w_l_tensor = (
        #    K.expand_dims(w_l, -1) * self.kernel_pos[None, None]
        #    + K.expand_dims(w_u, -1) * self.kernel_neg[None, None]
        # )
        # b_l_tensor = (
        #    K.expand_dims(b_l, -1) * self.kernel_pos[None]
        #    + K.expand_dims(b_u, -1) * self.kernel_neg[None]
        # )
        # l_tensor = (
        #    K.expand_dims(l_c, -1) * self.kernel_pos[None]
        #    + K.expand_dims(u_c, -1) * self.kernel_neg[None]
        # )

        # upper_grad = -grad_descent(
        #    x_0, -u_tensor, -w_u_tensor, -b_u_tensor, self.convex_domain, n_iter=30
        # )
        # lower_grad = grad_descent(
        #    x_0, l_tensor, w_l_tensor, b_l_tensor, self.convex_domain, n_iter=30
        # )

        if self.use_bias:
            b_u_ = K.bias_add(b_u_, self.bias, data_format="channels_last")
            b_l_ = K.bias_add(b_l_, self.bias, data_format="channels_last")
            u_c_ = K.bias_add(u_c_, self.bias, data_format="channels_last")
            l_c_ = K.bias_add(l_c_, self.bias, data_format="channels_last")
            y_ = K.bias_add(y_, self.bias, data_format="channels_last")

            # upper_grad = K.bias_add(upper_grad, self.bias, data_format="channels_last")
            # lower_grad = K.bias_add(lower_grad, self.bias, data_format="channels_last")

            if self.dc_decomp:
                h_ = K.bias_add(h_, K.maximum(0.0, self.bias), data_format="channels_last")
                g_ = K.bias_add(g_, K.minimum(0.0, self.bias), data_format="channels_last")

        upper_ = get_upper(x_0, w_u_, b_u_, self.convex_domain)
        u_c_ = K.minimum(upper_, u_c_)

        lower_ = get_lower(x_0, w_l_, b_l_, self.convex_domain)
        l_c_ = K.maximum(lower_, l_c_)

        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

        if self.dc_decomp:
            output += [h_, g_]

        if self.activation is not None:
            output = self.activation(
                output,
                dc_decomp=self.dc_decomp,
                grad_bounds=self.grad_bounds,
                convex_domain=self.convex_domain,
            )

        if self.dc_decomp:
            y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = output
        else:
            y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output

        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

        if self.dc_decomp:
            output += [h_, g_]

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
        # assert min([elem_0 == elem_1 for (elem_0, elem_1) in \
        #            zip(self.get_config(), dense.get_config())]), 'different configuration'

        if dense.built:
            params = dense.get_weights()
            w_pos = np.maximum(0.0, params[0])
            w_neg = np.minimum(0.0, params[0])
            if self.use_bias:
                self.set_weights([w_pos, w_neg, params[1]])
            else:
                self.set_weights([w_pos, w_neg])
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

    def __init__(self, **kwargs):
        """

        :param kwargs:

        """
        super(DecomonActivation, self).__init__(**kwargs)
        activation = kwargs["activation"]
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, input):
        return self.activation(input)


class DecomonFlatten(Flatten, DecomonLayer):
    def __init__(self, data_format=None, **kwargs):
        """

        :param data_format:
        :param kwargs:

        """
        super(DecomonFlatten, self).__init__(data_format=data_format, **kwargs)

        if self.grad_bounds:
            raise NotImplementedError()
        self.input_spec = [
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=2),
            InputSpec(min_ndim=2),
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=2),
            InputSpec(min_ndim=2),
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
        super(DecomonFlatten, self).build(y_input_shape)

        # return output_shape

    def call(self, inputs):

        if self.grad_bounds:
            raise NotImplementedError()

        op = super(DecomonFlatten, self).call

        if self.dc_decomp:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs
            h_ = op(h)
            g_ = op(g)
        else:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs

        y_ = op(y)
        b_u_ = op(b_u)
        b_l_ = op(b_l)
        u_c_ = op(u_c)
        l_c_ = op(l_c)

        output_shape = np.prod(list(K.int_shape(y_))[1:])
        input_dim = K.int_shape(x_0)[-1]

        w_u_ = K.reshape(w_u, (-1, input_dim, output_shape))
        w_l_ = K.reshape(w_l, (-1, input_dim, output_shape))

        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

        if self.dc_decomp:
            output += [h_, g_]

        return output


class DecomonReshape(Reshape, DecomonLayer):
    def __init__(self, target_shape, **kwargs):
        """

        :param data_format:
        :param kwargs:

        """
        super(DecomonReshape, self).__init__(target_shape=target_shape, **kwargs)
        if self.grad_bounds:
            raise NotImplementedError()
        self.input_spec = [
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=2),
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=1),
            InputSpec(min_ndim=2),
            InputSpec(min_ndim=1),
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
        # x_dim = input_shape[2]

        # import pdb; pdb.set_trace()
        # w_dim = [list(output_shape)[0], input_dim, list(output_shape)[1:]]

        # return [output_shape]*2+ [x_dim]*2 + [w_dim, output_shape]*2

    def call(self, inputs):

        if self.grad_bounds:
            raise NotImplementedError()

        op = super(DecomonReshape, self).call

        if self.dc_decomp:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs
            h_ = op(h)
            g_ = op(g)
        else:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs

        y_ = op(y)
        b_u_ = op(b_u)
        b_l_ = op(b_l)
        u_c_ = op(u_c)
        l_c_ = op(l_c)

        input_dim = K.int_shape(x_0)[-1]
        output_shape = [-1] + [input_dim] + list(self.target_shape)

        w_u_ = K.reshape(w_u, output_shape)
        w_l_ = K.reshape(w_l, output_shape)

        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

        if self.dc_decomp:
            output += [h_, g_]

        return output


def to_monotonic(layer, input_dim, dc_decomp=False, grad_bounds=False, convex_domain={}):
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
    layer_monotonic = globals()[monotonic_class_name].from_config(config_layer)

    input_shape = list(layer.input_shape)[1:]
    if isinstance(input_dim, tuple):
        x_shape = Input(input_dim)
        input_dim = input_dim[-1]
    else:
        x_shape = Input((input_dim,))

    w_shape = Input(tuple([input_dim] + input_shape))
    y_shape = Input(tuple(input_shape))

    input_ = [y_shape, x_shape, y_shape, w_shape, y_shape, y_shape, w_shape, y_shape]

    if dc_decomp:
        input_ += [y_shape, y_shape]

    layer_monotonic(input_)
    layer_monotonic.reset_layer(layer)

    return layer_monotonic


# Aliases
MonotonicConvolution2D = DecomonConv2D
