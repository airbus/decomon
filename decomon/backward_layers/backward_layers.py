from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten, Dot, Permute
from decomon.layers.decomon_layers import DecomonDense, DecomonConv2D
from ..backward_layers.activations import get
from tensorflow.keras.backend import conv2d
from .utils import V_slope


class BackwardDense(Layer):
    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardDense, self).__init__(**kwargs)
        if not isinstance(layer, DecomonDense):
            raise NotImplementedError()

        self.layer = layer
        self.activation = get(layer.get_config()["activation"])  # ??? not sur
        self.activation_name = layer.get_config()["activation"]
        self.slope = slope

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        # start with the activation: determine the upper and lower bounds before the weights
        weights = self.layer.kernel_pos + self.layer.kernel_neg
        bias = self.layer.bias

        # here update x
        x = self.layer.call_linear(x_)

        if self.activation_name != "linear":
            w_act_u, b_act_u, w_act_l, b_act_l = self.activation(
                x, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=self.layer.convex_domain, slope=self.slope
            )

            weights_u = w_act_u[:, None, :] * weights
            bias_u = K.sum(w_act_u[:, None, :] * bias[None] + b_act_u[:, None, :], 1)

            weights_l = w_act_l[:, None, :] * weights
            bias_l = K.sum(w_act_l[:, None, :] * bias[None] + b_act_l[:, None, :], 1)

            # weights_u = w_act_u[:,:, None]*weights
            # bias_u = K.sum(w_act_u[:,:, None]*bias[None] + b_act_u[:,:, None], 1)
            # weights_l = w_act_l[:, :, None] * weights
            # bias_l = K.sum(w_act_l[:, :, None] * bias[None] + b_act_l[:, :, None], 1)

        else:
            # import pdb; pdb.set_trace()
            weights_u = K.expand_dims(weights, 0) + K.zeros_like(w_out_u[:, 0, :1])
            bias_u = K.expand_dims(bias, 0) + K.zeros_like(K.max(b_out_u, 1))
            weights_l = K.expand_dims(weights, 0) + K.zeros_like(w_out_u[:, 0, :1])
            bias_l = K.expand_dims(bias, 0) + K.zeros_like(K.max(b_out_u, 1))

        op_perm = Permute((2, 1))
        # here

        w_out_u_ = K.sum(Dot(-2)([K.expand_dims(op_perm(weights_u), 1), K.maximum(w_out_u, 0)]), -2) + K.sum(
            Dot(-2)([K.expand_dims(op_perm(weights_l), 1), K.minimum(w_out_u, 0)]), -2
        )
        w_out_l_ = K.sum(Dot(-2)([K.expand_dims(op_perm(weights_l), 1), K.maximum(w_out_l, 0)]), -2) + K.sum(
            Dot(-2)([K.expand_dims(op_perm(weights_u), 1), K.minimum(w_out_l, 0)]), -2
        )

        b_out_u_ = (
            b_out_u
            + K.sum(Dot(-2)([K.maximum(w_out_u, 0), K.expand_dims(K.expand_dims(bias_u, 1), -1)]), (-2, -1))
            + K.sum(Dot(-2)([K.minimum(w_out_u, 0), K.expand_dims(K.expand_dims(bias_l, 1), -1)]), (-2, -1))
        )
        b_out_l_ = (
            b_out_l
            + K.sum(Dot(-2)([K.maximum(w_out_l, 0), K.expand_dims(K.expand_dims(bias_l, 1), -1)]), (-2, -1))
            + K.sum(Dot(-2)([K.minimum(w_out_l, 0), K.expand_dims(K.expand_dims(bias_u, 1), -1)]), (-2, -1))
        )

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

    def call(self, inputs):

        x = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        # start with the activation: determine the upper and lower bounds before the weights
        weights = self.layer.kernel_pos + self.layer.kernel_neg
        bias = self.layer.bias

        op_flat = Flatten()

        # do not do the activation so far
        # get input shape
        input_shape = x[0].shape[1:]
        n_dim = np.prod(input_shape)
        # import pdb; pdb.set_trace()
        # create diagonal matrix
        id_ = tf.linalg.diag(K.ones_like(op_flat(x[0][0:1])))[0]
        id_ = K.reshape(id_, tuple([n_dim] + list(input_shape)))

        weights = conv2d(
            id_,
            self.layer.kernel,
            strides=self.layer.strides,
            padding=self.layer.padding,
            data_format=self.layer.data_format,
            dilation_rate=self.layer.dilation_rate,
        )
        bias = self.layer.bias

        # flatten
        bias = K.zeros_like(weights) + bias[None, None, None]
        weights = op_flat(weights)
        bias = op_flat(bias)[0]
        # bias = K.reshape(bias, (-1,))

        # here update x
        x = self.layer.call_linear(x)

        if self.activation_name != "linear":

            w_act_u, b_act_u, w_act_l, b_act_l = self.activation(
                x, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=self.layer.convex_domain, slope=self.slope
            )
            w_act_u = op_flat(w_act_u)
            b_act_u = op_flat(b_act_u)
            w_act_l = op_flat(w_act_l)
            b_act_l = op_flat(b_act_l)

            weights_u = w_act_u[:, None, :] * weights
            bias_u = K.sum(w_act_u[:, None, :] * bias[None] + b_act_u[:, None, :], 1)

            weights_l = w_act_l[:, None, :] * weights
            bias_l = K.sum(w_act_l[:, None, :] * bias[None] + b_act_l[:, None, :], 1)

        else:

            # weights_u = K.expand_dims(weights, 0) + K.zeros_like(K.max(w_out_u, 1))
            # bias_u = K.expand_dims(bias, 0) + K.zeros_like(K.max(b_out_u, 1))
            # weights_l = K.expand_dims(weights, 0)  + K.zeros_like(K.max(w_out_u, 1))
            # bias_l = K.expand_dims(bias, 0) + K.zeros_like(K.max(b_out_u, 1))
            weights_u = K.expand_dims(weights, 0) + K.zeros_like(w_out_u[:, 0, :1])
            weights_l = K.expand_dims(weights, 0) + K.zeros_like(w_out_u[:, 0, :1])
            tmp = op_flat(K.zeros_like(b_out_u))
            bias_u = K.expand_dims(bias, 0) + tmp[:, 0:1]
            bias_l = K.expand_dims(bias, 0) + tmp[:, 0:1]

        op_perm = Permute((2, 1))
        # here
        w_out_u_ = K.sum(Dot(-2)([K.expand_dims(op_perm(weights_u), 1), K.maximum(w_out_u, 0)]), -2) + K.sum(
            Dot(-2)([K.expand_dims(op_perm(weights_l), 1), K.minimum(w_out_u, 0)]), -2
        )
        w_out_l_ = K.sum(Dot(-2)([K.expand_dims(op_perm(weights_l), 1), K.maximum(w_out_l, 0)]), -2) + K.sum(
            Dot(-2)([K.expand_dims(op_perm(weights_u), 1), K.minimum(w_out_l, 0)]), -2
        )

        b_out_u_ = (
            b_out_u
            + K.sum(Dot(-2)([K.maximum(w_out_u, 0), K.expand_dims(K.expand_dims(bias_u, 1), -1)]), (-2, -1))
            + K.sum(Dot(-2)([K.minimum(w_out_u, 0), K.expand_dims(K.expand_dims(bias_l, 1), -1)]), (-2, -1))
        )
        b_out_l_ = (
            b_out_l
            + K.sum(Dot(-2)([K.maximum(w_out_l, 0), K.expand_dims(K.expand_dims(bias_l, 1), -1)]), (-2, -1))
            + K.sum(Dot(-2)([K.minimum(w_out_l, 0), K.expand_dims(K.expand_dims(bias_u, 1), -1)]), (-2, -1))
        )

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_


class BackwardFlatten(Layer):
    def call(self, inputs, slope=V_slope.name):

        return inputs[-4:]


class BackwardReshape(Layer):
    def call(self, inputs, slope=V_slope.name):
        return inputs[-4:]


# TO DO: decomonActivation and MaxPooling


def get_backward(layer, slope=V_slope.name, **kwargs):

    # do it better
    class_name = "".join(layer.__class__.__name__.split("Decomon")[1:])

    backward_class_name = "Backward{}".format(class_name)
    class_ = globals()[backward_class_name]

    return class_(layer, slope=slope)
