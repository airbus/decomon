from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten, Dot, Permute
from decomon.layers.decomon_layers import (
    DecomonDense,
    DecomonConv2D,
    DecomonDropout,
    DecomonReshape,
    DecomonFlatten,
    DecomonBatchNormalization,
)
from ..backward_layers.activations import get
from tensorflow.keras.backend import conv2d, conv2d_transpose
from .utils import V_slope
from .backward_maxpooling import BackwardMaxPooling2D
from tensorflow.python.ops import array_ops


class BackwardDense(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardDense, self).__init__(**kwargs)
        if not isinstance(layer, DecomonDense):
            raise NotImplementedError()

        self.layer = layer
        self.activation = get(layer.get_config()["activation"])  # ??? not sur
        self.activation_name = layer.get_config()["activation"]
        self.slope = slope
        self.mode = self.layer.mode

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        # start with the activation: determine the upper and lower bounds before the weights
        weights = self.layer.kernel
        if self.layer.use_bias:
            bias = self.layer.bias

        if self.activation_name != "linear":
            # here update x
            x = self.layer.call_linear(x_)
            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x,
                w_out_u,
                b_out_u,
                w_out_l,
                b_out_l,
                convex_domain=self.layer.convex_domain,
                slope=self.slope,
                mode=self.mode,
            )

        weights = K.expand_dims(K.expand_dims(K.expand_dims(weights, 0), 1), -1)
        if self.layer.use_bias:
            bias = K.expand_dims(K.expand_dims(K.expand_dims(bias, 0), 1), -1)
            b_out_u_ = K.sum(w_out_u * bias, 2) + b_out_u
            b_out_l_ = K.sum(w_out_l * bias, 2) + b_out_l
        else:
            b_out_u_ = b_out_u
            b_out_l_ = b_out_l

        w_out_u = K.expand_dims(w_out_u, 2)
        w_out_l = K.expand_dims(w_out_l, 2)
        w_out_u_ = K.sum(w_out_u * weights, 3)
        w_out_l_ = K.sum(w_out_l * weights, 3)

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_


class BackwardConv2D(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardConv2D, self).__init__(**kwargs)
        if not isinstance(layer, DecomonConv2D):
            raise NotImplementedError()

        self.layer = layer
        self.activation = get(layer.get_config()["activation"])  # ??? not sur
        self.activation_name = layer.get_config()["activation"]
        self.slope = slope
        self.mode = self.layer.mode

    def call(self, inputs):

        x = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        # infer the output dimension
        x_output = self.layer.call_linear(x)

        if self.activation_name != "linear":
            # here update x

            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x_output,
                w_out_u,
                b_out_u,
                w_out_l,
                b_out_l,
                convex_domain=self.layer.convex_domain,
                slope=self.slope,
                mode=self.mode,
            )

        output_shape_tensor = x_output[0].shape
        n_out = w_out_u.shape[-1]
        # first permute dimensions
        w_out_u = array_ops.transpose(w_out_u, perm=(0, 1, 3, 2))
        w_out_l = array_ops.transpose(w_out_l, perm=(0, 1, 3, 2))

        if self.layer.data_format == "channels_last":
            height, width, channel = [e for e in output_shape_tensor[1:]]
            w_out_u = K.reshape(w_out_u, [-1, n_out, height, width, channel])
            w_out_l = K.reshape(w_out_l, [-1, n_out, height, width, channel])
        else:
            channel, height, width = [e for e in output_shape_tensor[1:]]
            w_out_u = K.reshape(w_out_u, [-1, n_out, channel, height, width, channel])
            w_out_l = K.reshape(w_out_l, [-1, n_out, channel, height, width, channel])

        # start with bias
        if self.layer.use_bias:
            bias = self.layer.bias[None, None, None, None]
            b_out_u_ = K.expand_dims(K.sum(w_out_u * bias, (2, 3, 4)), 1) + b_out_u
            b_out_l_ = K.expand_dims(K.sum(w_out_l * bias, (2, 3, 4)), 1) + b_out_l
        else:
            b_out_u_ = K.expand_dims(K.sum(w_out_u * bias, (2, 3, 4)), 1)
            b_out_l_ = K.expand_dims(K.sum(w_out_l * bias, (2, 3, 4)), 1)

        kernel_ = array_ops.transpose(self.layer.kernel, (0, 1, 3, 2))

        def step_func(z, _):
            return (
                conv2d_transpose(
                    z,
                    self.layer.kernel,
                    x[0].shape,
                    strides=self.layer.strides,
                    padding=self.layer.padding,
                    data_format=self.layer.data_format,
                    dilation_rate=self.layer.dilation_rate,
                ),
                [],
            )

        # import pdb; pdb.set_trace()
        step_func(w_out_u[:, 0], 0)
        w_out_u_ = K.rnn(step_function=step_func, inputs=w_out_u, initial_states=[], unroll=False)[1]
        w_out_l_ = K.rnn(step_function=step_func, inputs=w_out_l, initial_states=[], unroll=False)[1]

        n_in = np.prod(w_out_u_.shape[2:])
        w_out_u_ = array_ops.transpose(K.reshape(w_out_u_, [-1, 1, n_out, n_in]), perm=(0, 1, 3, 2))
        w_out_l_ = array_ops.transpose(K.reshape(w_out_l_, [-1, 1, n_out, n_in]), perm=(0, 1, 3, 2))

        """
        op_flat = Flatten()
    

        # do not do the activation so far
        # get input shape
        input_shape = x[0].shape[1:]
        n_dim = np.prod(input_shape)
        # import pdb; pdb.set_trace()
        # create diagonal matrix
        id_ = tf.linalg.diag(1.+0*(op_flat(x[0][0:1])))[0]
        id_ = K.reshape(id_, tuple([n_dim] + list(input_shape)))

        id_ = K.expand_dims(id_, 0)

        def step_func(x, _):
            return conv2d(
                x,
                self.layer.kernel,
                strides=self.layer.strides,
                padding=self.layer.padding,
                data_format=self.layer.data_format,
                dilation_rate=self.layer.dilation_rate), []



        weights = K.rnn(step_function=step_func, inputs=id_, initial_states=[], unroll=False)[1]

        bias = self.layer.bias


        # flatten
        n_in = weights.shape[1]
        n_layer = [e for e in weights.shape[2:]]
        n_out = w_out_u.shape[-1]
        weights = K.reshape(weights, (-1, n_in, np.prod(n_layer), 1))

        # bias = K.reshape(bias, (-1,))

        if self.activation_name != "linear":
            # here update x
            x = self.layer.call_linear(x)

            w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                x,
                w_out_u,
                b_out_u,
                w_out_l,
                b_out_l,
                convex_domain=self.layer.convex_domain,
                slope=self.slope,
                mode=self.mode,
            )



        #weights = K.expand_dims(weights, -1)
        #bias = K.expand_dims(K.expand_dims(K.expand_dims(bias, -1), 0), 0)


        w_out_u_ = K.expand_dims(K.sum(w_out_u*weights, 2), 1)
        w_out_l_ = K.expand_dims(K.sum(w_out_l * weights, 2), 1)
        w_u_0 = K.reshape(w_out_u, [-1]+n_layer+[n_out])
        w_l_0 = K.reshape(w_out_l, [-1] + n_layer + [n_out])
        bias = bias[None, None, None, :, None]
        b_out_u_ = K.expand_dims(K.sum(w_u_0*bias, (1, 2, 3)), 1) + b_out_u
        b_out_l_ = K.expand_dims(K.sum(w_l_0 * bias, (1, 2, 3)), 1) + b_out_l
        #b_out_u_ = K.sum(w_out_u*bias, 2) + b_out_u
        #b_out_l_ = K.sum(w_out_l * bias, 2) + b_out_l
        """

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_


class BackwardFlatten(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardFlatten, self).__init__(**kwargs)
        if not isinstance(layer, DecomonFlatten):
            raise NotImplementedError()

    def call(self, inputs, slope=V_slope.name):

        return inputs[-4:]


class BackwardReshape(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardReshape, self).__init__(**kwargs)
        if not isinstance(layer, DecomonReshape):
            raise NotImplementedError()

    def call(self, inputs, slope=V_slope.name):
        return inputs[-4:]


class BackwardDropout(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardDropout, self).__init__(**kwargs)
        if not isinstance(layer, DecomonDropout):
            raise NotImplementedError()

    def call(self, inputs, slope=V_slope.name):
        return inputs[-4:]


class BackwardBatchNormalization(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardBatchNormalization, self).__init__(**kwargs)
        if not isinstance(layer, DecomonBatchNormalization):
            raise NotImplementedError()
        self.layer = layer
        self.mode = self.layer.mode
        self.axis = self.layer.axis
        self.op_flat = Flatten()

    def call(self, inputs, slope=V_slope.name):

        y = inputs[0]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_dim = y.shape[1:]
        n_out = w_out_u.shape[-1]
        # reshape
        w_out_u = K.reshape(w_out_u, [-1, 1] + list(n_dim) + [n_out])
        w_out_l = K.reshape(w_out_l, [-1, 1] + list(n_dim) + [n_out])

        n_dim = len(y.shape)
        tuple_ = [1] * n_dim
        for i, ax in enumerate(self.axis):
            tuple_[ax] = self.layer.moving_mean.shape[i]

        gamma_ = K.reshape(self.layer.gamma + 0.0, tuple_)
        beta_ = K.reshape(self.layer.beta + 0.0, tuple_)
        moving_mean_ = K.reshape(self.layer.moving_mean + 0.0, tuple_)
        moving_variance_ = K.reshape(self.layer.moving_variance + 0.0, tuple_)

        w_ = gamma_ / K.sqrt(moving_variance_ + self.layer.epsilon)
        b_ = beta_ - w_ * moving_mean_

        # flatten w_, b_
        # w_ = self.op_flat(w_)
        # b_ = self.op_flat(b_) - w_ * self.op_flat(moving_mean_)

        w_ = K.expand_dims(K.expand_dims(w_, -1), 1)
        b_ = K.expand_dims(K.expand_dims(b_, -1), 1)

        n_dim = np.prod(y.shape[1:])
        w_u_b_ = K.reshape(w_out_u * w_, (-1, n_dim, n_out))
        w_l_b_ = K.reshape(w_out_l * w_, (-1, n_dim, n_out))
        axis = [i for i in range(2, len(b_.shape) - 1)]
        b_u_b_ = K.sum(w_out_u * b_, axis) + b_out_u
        b_l_b_ = K.sum(w_out_l * b_, axis) + b_out_l

        return w_u_b_, b_u_b_, w_l_b_, b_l_b_


# TO DO: decomonActivation and MaxPooling


def get_backward(layer, slope=V_slope.name, **kwargs):

    # do it better
    class_name = "".join(layer.__class__.__name__.split("Decomon")[1:])

    backward_class_name = "Backward{}".format(class_name)
    class_ = globals()[backward_class_name]

    try:

        return class_(layer, slope=slope)
    except TypeError:
        import pdb

        pdb.set_trace()
