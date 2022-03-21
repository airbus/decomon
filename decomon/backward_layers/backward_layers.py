from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten, Permute, Activation
from decomon.layers.decomon_layers import (
    DecomonDense,
    DecomonConv2D,
    DecomonDropout,
    DecomonReshape,
    DecomonFlatten,
    DecomonBatchNormalization,
    DecomonActivation,
    DecomonPermute,
)
from ..backward_layers.activations import get
from tensorflow.keras.backend import conv2d, conv2d_transpose
from .utils import V_slope, backward_sort, get_identity_lirpa, get_IBP, get_FORWARD, get_input_dim
from ..layers.utils import ClipAlpha, F_HYBRID, F_FORWARD
from ..layers.decomon_layers import to_monotonic
from .backward_maxpooling import BackwardMaxPooling2D
from .backward_merge import BackwardAverage
from tensorflow.python.ops import array_ops


class BackwardDense(Layer):
    """
    Backward  LiRPA of Dense
    """

    def __init__(
        self,
        layer,
        slope=V_slope.name,
        previous=True,
        mode=F_HYBRID.name,
        convex_domain={},
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super(BackwardDense, self).__init__(**kwargs)

        self.layer = layer
        self.activation = get(layer.get_config()["activation"])  # ??? not sur
        self.activation_name = layer.get_config()["activation"]
        self.slope = slope
        # if hasattr(self.layer, 'finetune'):
        #    self.finetune=self.layer.finetune
        # else:
        #    self.finetune = False
        self.finetune = finetune
        self.previous = previous
        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = mode
            self.convex_domain = convex_domain
            input_dim_ = get_input_dim(input_dim, self.convex_domain)
            self.layer = to_monotonic(
                layer,
                input_dim_,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                IBP=get_IBP(self.mode),
                forward=get_FORWARD(self.mode),
                shared=True,
                fast=False,
            )[0]

        self.frozen_weights = False

    def call_previous(self, inputs):

        if not len(inputs):
            inputs = self.layer.input

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        # start with the activation: determine the upper and lower bounds before the weights
        weights = self.layer.kernel
        if self.layer.use_bias:
            bias = self.layer.bias

        if self.activation_name != "linear":
            # here update x
            x = self.layer.call_linear(x_)
            if self.finetune:
                if self.activation_name[:4] != "relu":
                    w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                        x + [w_out_u, b_out_u, w_out_l, b_out_l],
                        convex_domain=self.convex_domain,
                        slope=self.slope,
                        mode=self.mode,
                        previous=self.previous,
                        finetune=[self.alpha_b_u, self.alpha_b_l],
                    )
                else:
                    w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                        x + [w_out_u, b_out_u, w_out_l, b_out_l],
                        convex_domain=self.convex_domain,
                        slope=self.slope,
                        mode=self.mode,
                        previous=self.previous,
                        finetune=self.alpha_b_l,
                    )
            else:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    x + [w_out_u, b_out_u, w_out_l, b_out_l],
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                )
        # w_out_u (None, n_out, n_back)  b_out_u (None, n_back)
        weights = K.expand_dims(K.expand_dims(weights, 0), -1)  # (1, n_in, n_out, 1)
        if self.layer.use_bias:
            bias = K.expand_dims(K.expand_dims(bias, 0), -1)  # (None, n_out, 1)
            b_out_u_ = K.sum(w_out_u * bias, 1) + b_out_u  # (None, n_back)
            b_out_l_ = K.sum(w_out_l * bias, 1) + b_out_l
        else:
            b_out_u_ = b_out_u
            b_out_l_ = b_out_l
        w_out_u = K.expand_dims(w_out_u, 1)
        w_out_l = K.expand_dims(w_out_l, 1)
        w_out_u_ = K.sum(w_out_u * weights, 2)  # (None, n_in,  n_back)
        w_out_l_ = K.sum(w_out_l * weights, 2)

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_

    def call_no_previous(self, inputs):

        if len(inputs):
            x_ = inputs
        else:
            x_ = self.layer.input

        # start with the activation: determine the upper and lower bounds before the weights
        weights = self.layer.kernel
        if self.layer.use_bias:
            bias = self.layer.bias

        if self.activation_name != "linear":
            # here update x
            x = self.layer.call_linear(x_)
            if self.finetune:
                if self.activation_name[:4] != "relu":
                    w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                        x,
                        convex_domain=self.convex_domain,
                        slope=self.slope,
                        mode=self.mode,
                        previous=self.previous,
                        finetune=[self.alpha_b_u, self.alpha_b_l],
                    )
                else:
                    w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                        x,
                        convex_domain=self.convex_domain,
                        slope=self.slope,
                        mode=self.mode,
                        previous=self.previous,
                        finetune=self.alpha_b_l,
                    )
            else:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    x,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    previous=self.previous,
                    mode=self.mode,
                )
            # w_out_u (None, n_out)  b_out_u (None, n_back)
            w_out_u = tf.linalg.diag(w_out_u)  # (None, n_out, n_back=n_out)
            w_out_l = tf.linalg.diag(w_out_l)
            weights = K.expand_dims(K.expand_dims(weights, 0), -1)  # (1, n_in, n_out, 1)
            if self.layer.use_bias:
                bias = K.expand_dims(K.expand_dims(bias, 0), -1)  # (None, n_out, 1)
                b_out_u_ = K.sum(w_out_u * bias, 1) + b_out_u  # (None, n_back)
                b_out_l_ = K.sum(w_out_l * bias, 1) + b_out_l
            else:
                b_out_u_ = b_out_u
                b_out_l_ = b_out_l
            w_out_u = K.expand_dims(w_out_u, 1)  # (None,  1, n_in, n_back)
            w_out_l = K.expand_dims(w_out_l, 1)
            w_out_u_ = K.sum(w_out_u * weights, 2)  # (None, n_in,  n_back)
            w_out_l_ = K.sum(w_out_l * weights, 2)
        else:
            y_ = x_[-1]
            z_value = K.cast(0.0, K.floatx())

            w_out_u_, w_out_l_ = [weights[None] + z_value * K.expand_dims(y_, -1)] * 2
            if self.layer.use_bias:
                b_out_u_, b_out_l_ = [bias[None] + z_value * w_out_u_[:, 0]] * 2  # not sure....
            else:
                b_out_u_, b_out_l_ = [z_value * w_out_u_[:, 0]] * 2

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_

    def call(self, inputs):
        if self.previous:
            return self.call_previous(inputs)
        else:
            return self.call_no_previous(inputs)

    def build(self, input_shape):
        """
        :param input_shape: list of input shape
        :return:
        """

        if self.finetune and self.activation_name != "linear":
            units = self.layer.units
            self.alpha_b_l = self.add_weight(
                shape=(units,), initializer="ones", name="alpha_l_b", regularizer=None, constraint=ClipAlpha()
            )

            if self.activation_name[:4] != "relu":
                self.alpha_b_u = self.add_weight(
                    shape=(units,), initializer="ones", name="alpha_u_b", regularizer=None, constraint=ClipAlpha()
                )
        self.built = True

    def freeze_alpha(self):
        if self.finetune:
            self.trainable = False

    def unfreeze_alpha(self):
        if self.finetune:
            self.trainable = True

    def reset_finetuning(self):
        if self.finetune and self.activation_name != "linear":
            K.set_value(self.alpha_b_l, np.ones_like(self.alpha_b_l.value()))

            if self.activation_name[:4] != "relu":
                K.set_value(self.alpha_b_u, np.ones_like(self.alpha_b_u.value()))

    def freeze_weights(self):

        if not self.frozen_weights:

            if self.finetune and self.mode == F_HYBRID.name:
                if self.layer.use_bias:
                    self._trainable_weights = self._trainable_weights[2:]
                else:
                    self._trainable_weights = self._trainable_weights[1:]
            else:
                self._trainable_weights = []

            if getattr(self.layer, "freeze_weights"):
                self.layer.freeze_weights()

            self.frozen_weights = True

    def unfreeze_weights(self):
        if self.frozen_weights:
            if getattr(self.layer, "unfreeze_weights"):
                self.layer.unfreeze_weights()
            self.frozen_weights = False


class BackwardConv2D(Layer):
    """
    Backward  LiRPA of Conv2D
    """

    def __init__(
        self,
        layer,
        slope=V_slope.name,
        previous=True,
        mode=F_HYBRID.name,
        convex_domain={},
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super(BackwardConv2D, self).__init__(**kwargs)

        self.layer = layer
        self.activation = get(layer.get_config()["activation"])  # ??? not sur
        self.activation_name = layer.get_config()["activation"]
        self.slope = slope
        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = mode
            self.convex_domain = convex_domain
        # if hasattr(self.layer, 'finetune'):
        #    self.finetune = self.layer.finetune
        # else:
        #    self.finetune=finetune
        self.finetune = finetune

        self.previous = previous

        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = mode
            self.convex_domain = convex_domain
            input_dim_ = get_input_dim(input_dim, self.convex_domain)
            self.layer = to_monotonic(
                layer,
                input_dim_,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                IBP=get_IBP(self.mode),
                forward=get_FORWARD(self.mode),
                shared=True,
                fast=False,
            )[0]

        self.frozen_weights = False

    def get_bounds_linear(self, w_out_u, b_out_u, w_out_l, b_out_l ):

        output_shape_tensor = self.layer.output_shape[-1]
        n_out = w_out_u.shape[-1]
        # first permute dimensions
        w_out_u = array_ops.transpose(w_out_u, perm=(0, 2, 1))
        w_out_l = array_ops.transpose(w_out_l, perm=(0, 2, 1))

        if self.layer.data_format == "channels_last":
            height, width, channel = [e for e in output_shape_tensor[1:]]
            w_out_u = K.reshape(w_out_u, [-1, n_out, height, width, channel])
            w_out_l = K.reshape(w_out_l, [-1, n_out, height, width, channel])

            # start with bias
            if self.layer.use_bias:
                bias = self.layer.bias[None, None, None, None]  # (1, 1, 1, 1, channel)
                # w_out_u * bias : (None, n_out, height, width, channel)
                b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
                b_out_u_ += b_out_u
                b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
                b_out_l_ += b_out_l
            else:
                b_out_u_ = b_out_u
                b_out_l_ = b_out_l

        else:
            channel, height, width = [e for e in output_shape_tensor[1:]]
            w_out_u = K.reshape(w_out_u, [-1, n_out, channel, height, width])
            w_out_l = K.reshape(w_out_l, [-1, n_out, channel, height, width])

            # start with bias
            if self.layer.use_bias:
                bias = self.layer.bias[None, None, :, None, None]  # (1, 1, channel, 1, 1)
                # w_out_u * bias : (None, n_out, height, width, channel)
                b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
                b_out_u_ += b_out_u
                b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
                b_out_l_ += b_out_l
            else:
                b_out_u_ = b_out_u
                b_out_l_ = b_out_l

        def step_func(z, _):
            return (
                conv2d_transpose(
                    z,
                    self.layer.kernel,
                    self.layer.get_input_shape_at(0)[-1],
                    strides=self.layer.strides,
                    padding=self.layer.padding,
                    data_format=self.layer.data_format,
                    dilation_rate=self.layer.dilation_rate,
                ),
                [],
            )

        step_func(w_out_u[:, 0], 0)  # init
        w_out_u_ = K.rnn(step_function=step_func, inputs=w_out_u, initial_states=[], unroll=False)[1]
        w_out_l_ = K.rnn(step_function=step_func, inputs=w_out_l, initial_states=[], unroll=False)[1]

        n_in = np.prod(w_out_u_.shape[2:])
        w_out_u_ = array_ops.transpose(K.reshape(w_out_u_, [-1, n_out, n_in]), perm=(0, 2, 1))
        w_out_l_ = array_ops.transpose(K.reshape(w_out_l_, [-1, n_out, n_in]), perm=(0, 2, 1))

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_


    def call_previous(self, inputs):
        x = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]


        if self.activation_name != "linear":
            x_output = self.layer.call_linear(x)
            # here update x
            if self.finetune:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    x_output + [w_out_u, b_out_u, w_out_l, b_out_l],
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune=self.alpha_b_l,
                )
            else:

                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    x_output + [w_out_u, b_out_u, w_out_l, b_out_l],
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                )
        return self.get_bounds_linear(w_out_u, b_out_u, w_out_l, b_out_l )

    def call_no_previous(self, inputs):
        x = inputs


        if self.activation_name != "linear":
            # here update x
            x_output = self.layer.call_linear(x)
            y_ = x_output[-1]
            shape = np.prod(y_.shape[1:])

            if self.finetune:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    x_output,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune=self.alpha_b_l,
                    previous=self.previous,
                )

            else:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    x_output, convex_domain=self.convex_domain, slope=self.slope, mode=self.mode, previous=self.previous
                )
                w_out_u = K.reshape(w_out_u, (-1, shape))
                b_out_u = K.reshape(b_out_u, (-1, shape))
                w_out_l = K.reshape(w_out_l, (-1, shape))
                b_out_l = K.reshape(b_out_l, (-1, shape))

            return self.get_bounds_linear(w_out_u, b_out_u, w_out_l, b_out_l)

        else:
            weight, bias = self.layer.get_backward_weights(x)

            z_value = K.cast(0.0, K.floatx())
            y_ = x[-1]
            shape = np.prod(y_.shape[1:])
            y_flatten = K.reshape(z_value * y_, (-1, np.prod(shape), 1))  # (None, n_in, 1)
            w_out_u_ = y_flatten + K.expand_dims(weight, 0)
            w_out_l_ = w_out_u_
            b_out_u_ = K.sum(y_flatten, 1) + K.expand_dims(bias, 0)
            b_out_l_ = b_out_u_

        return w_out_u_, b_out_u_, w_out_l_, b_out_l_

    def call(self, inputs):

        if self.previous:
            return self.call_previous(inputs)
        else:
            return self.call_no_previous(inputs)

    def build(self, input_shape):
        """
        :param input_shape: list of input shape
        :return:
        """

        if self.finetune and self.activation_name != "linear":
            output_shape = self.layer.compute_output_shape(input_shape[:-4])[0]

            units = np.prod(output_shape[1:])
            self.alpha_b_l = self.add_weight(
                shape=(units,), initializer="ones", name="alpha_l_b", regularizer=None, constraint=ClipAlpha()
            )

        self.built = True

    def freeze_alpha(self):
        if self.finetune:
            self.trainable = False

    def unfreeze_alpha(self):
        if self.finetune:
            self.trainable = True

    def reset_finetuning(self):
        if self.finetune and self.activation_name != "linear":

            K.set_value(self.alpha_b_l, np.ones_like(self.alpha_b_l.value()))

    def freeze_weights(self):

        if not self.frozen_weights:

            if self.finetune and self.mode == F_HYBRID.name:
                if self.layer.use_bias:
                    self._trainable_weights = self._trainable_weights[2:]
                else:
                    self._trainable_weights = self._trainable_weights[1:]
            else:
                self._trainable_weights = []

            if getattr(self.layer, "freeze_weights"):
                self.layer.freeze_weights()

            self.frozen_weights = True

    def unfreeze_weights(self):
        if self.frozen_weights:
            if self.finetune and self.mode == F_HYBRID.name:
                if getattr(self.layer, "unfreeze_weights"):
                    self.layer.unfreeze_weights()
            self.frozen_weights = False


class BackwardActivation(Layer):
    def __init__(
        self, layer, slope=V_slope.name, previous=True, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs
    ):
        super(BackwardActivation, self).__init__(**kwargs)

        self.layer = layer
        self.activation = get(layer.get_config()["activation"])  # ??? not sur
        self.activation_name = layer.get_config()["activation"]
        self.slope = slope
        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = mode
            self.convex_domain = convex_domain
        self.previous = previous
        self.finetune = finetune
        self.finetune_param = []
        if self.finetune:
            self.frozen_alpha=False

    def build(self, input_shape):
        """
        :param input_shape: list of input shape
        :return:
        """

        if self.finetune and self.activation_name != "linear":

            if self.previous:
                input_dim = np.prod(input_shape[-5][1:])
            else:
                input_dim = np.prod(input_shape[-1][1:])
            self.alpha_b_l = self.add_weight(
                shape=(input_dim,), initializer="ones", name="alpha_l_b", regularizer=None, constraint=ClipAlpha()
            )

            if self.activation_name[:4] != "relu":
                self.alpha_b_u = self.add_weight(
                    shape=(input_dim,), initializer="ones", name="alpha_u_b", regularizer=None, constraint=ClipAlpha()
                )
                self.finetune_param.append(self.alpha_b_u)

            self.finetune_param.append(self.alpha_b_l)
            if len(self.finetune_param) == 1:
                self.finetune_param = self.finetune_param[0]

        self.built = True

    def call_previous(self, inputs):
        x = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        # infer the output dimension
        if self.finetune:
            if self.activation_name != "linear":
                # here update x

                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune=self.finetune_param,
                )
        else:
            if self.activation_name != "linear":
                # here update x

                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                )
        # reshape
        # shape = np.prod(x[-1].shape[1:])
        # op_reshape = Reshape((shape, -1))
        # w_out_u = op_reshape(w_out_u)
        # w_out_l = op_reshape(w_out_l)

        return w_out_u, b_out_u, w_out_l, b_out_l

    def call_no_previous(self, inputs):

        # infer the output dimension
        y_ = inputs[-1]
        shape = np.prod(y_.shape[1:])

        if self.activation_name != "linear":
            # here update x
            if self.finetune:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    previous=self.previous,  # add hyperparameters
                    finetune=self.finetune_param,
                )
            else:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    previous=self.previous,  # add hyperparameters
                )
        else:
            y_ = inputs[-1]
            shape = np.prod(y_.shape[1:])

            z_value = K.cast(0.0, K.floatx())
            o_value = K.cast(1.0, K.floatx())
            y_flat = K.reshape(y_, [-1, shape])

            w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
            b_out_u, b_out_l = [z_value * y_flat] * 2

        w_out_u = tf.linalg.diag(w_out_u)
        w_out_l = tf.linalg.diag(w_out_l)

        return w_out_u, b_out_u, w_out_l, b_out_l

    def call(self, inputs):
        if self.previous:
            y_ = inputs[:-4][-1]
            w_out_u, w_out_l, b_out_u, b_out_l = self.call_previous(inputs)
        else:
            y_ = inputs[-1]

            w_out_u, w_out_l, b_out_u, b_out_l = self.call_no_previous(inputs)

        return w_out_u, w_out_l, b_out_u, b_out_l

    def freeze_alpha(self):
        if not self.frozen_alpha:
            if self.finetune and self.mode in [F_FORWARD.name,F_HYBRID.name]:
                self._trainable_weights = []
                self.frozen_alpha = True


    def unfreeze_alpha(self):
        if self.frozen_alpha:
            if self.finetune and self.mode in [F_FORWARD.name, F_HYBRID.name]:
                if self.activation_name!='linear':
                    if self.activation_name[:4]!='relu':
                        self._trainable_weights += [self.alpha_b_u, self.alpha_b_l]
                    else:
                        self._trainable_weights += [self.alpha_b_l]
            self.frozen_alpha = False


class BackwardFlatten(Layer):
    """
    Backward  LiRPA of Flatten
    """

    def __init__(
        self, layer, slope=V_slope.name, previous=True, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs
    ):
        super(BackwardFlatten, self).__init__(**kwargs)
        self.previous = previous

    def call(self, inputs, slope=V_slope.name):
        if self.previous:
            return inputs[-4:]
        else:
            y_ = inputs[-1]
            shape = np.prod(y_.shape[1:])

            z_value = K.cast(0.0, K.floatx())
            o_value = K.cast(1.0, K.floatx())
            y_flat = K.reshape(y_, [-1, shape])

            w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
            b_out_u, b_out_l = [z_value * y_flat] * 2
            w_out_u = tf.linalg.diag(w_out_u)
            w_out_l = tf.linalg.diag(w_out_l)

            return w_out_u, b_out_u, w_out_l, b_out_l


class BackwardReshape(Layer):
    """
    Backward  LiRPA of Reshape
    """

    def __init__(
        self, layer, slope=V_slope.name, previous=True, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs
    ):
        super(BackwardReshape, self).__init__(**kwargs)
        self.previous = previous

    def call(self, inputs, slope=V_slope.name):
        if self.previous:
            w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
        else:
            y_ = inputs[-1]
            shape = np.prod(y_.shape[1:])

            z_value = K.cast(0.0, K.floatx())
            o_value = K.cast(1.0, K.floatx())
            y_flat = K.reshape(y_, [-1, shape])

            w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
            b_out_u, b_out_l = [z_value * y_flat] * 2
            w_out_u = tf.linalg.diag(w_out_u)
            w_out_l = tf.linalg.diag(w_out_l)

        return w_out_u, b_out_u, w_out_l, b_out_l


class BackwardPermute(Layer):
    """
    Backward LiRPA of Permute
    """

    def __init__(
        self, layer, slope=V_slope.name, previous=True, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs
    ):
        super(BackwardPermute, self).__init__(**kwargs)
        self.dims = layer.dims
        self.op = layer.call
        self.previous = previous

    def call(self, inputs, slope=V_slope.name):

        if self.previous:
            w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
            y = inputs[:-4][-1]
        else:
            y = inputs[-1]
            shape = np.prod(y.shape[1:])
            z_value = K.cast(0.0, K.floatx())
            o_value = K.cast(1.0, K.floatx())
            y_flat = K.reshape(y, [-1, shape])

            w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
            b_out_u, b_out_l = [z_value * y_flat] * 2
            w_out_u = tf.linalg.diag(w_out_u)
            w_out_l = tf.linalg.diag(w_out_l)

        # w_out_u (None, n_in, n_out)

        n_dim = w_out_u.shape[1]
        n_out = w_out_u.shape[-1]
        shape = list(y.shape[1:])

        w_out_u_ = K.reshape(w_out_u, [-1] + shape + [n_out])
        w_out_l_ = K.reshape(w_out_l, [-1] + shape + [n_out])

        dims = [0] + list(self.dims) + [len(y.shape)]
        dims = list(np.argsort(dims))
        w_out_u_0 = K.reshape(K.permute_dimensions(w_out_u_, dims), (-1, n_dim, n_out))
        w_out_l_0 = K.reshape(K.permute_dimensions(w_out_l_, dims), (-1, n_dim, n_out))

        return [w_out_u_0, b_out_u, w_out_l_0, b_out_l]


class BackwardDropout(Layer):
    """
    Backward  LiRPA of Dropout
    """

    def __init__(
        self, layer, slope=V_slope.name, previous=True, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs
    ):
        super(BackwardDropout, self).__init__(**kwargs)

    def call(self, inputs, slope=V_slope.name):

        if self.previous:
            w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
        else:
            y_ = inputs[-1]
            shape = np.prod(y_.shape[1:])

            z_value = K.cast(0.0, K.floatx())
            o_value = K.cast(1.0, K.floatx())
            y_flat = K.reshape(y_, [-1, shape])

            w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
            b_out_u, b_out_l = [z_value * y_flat] * 2
            w_out_u = tf.linalg.diag(w_out_u)
            w_out_l = tf.linalg.diag(w_out_l)

        return w_out_u, b_out_u, w_out_l, b_out_l


class BackwardBatchNormalization(Layer):
    """
    Backward  LiRPA of Batch Normalization
    """

    def __init__(self, layer, slope=V_slope.name, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs):
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


class BackwardInputLayer(Layer):
    def __init__(
        self, layer, slope=V_slope.name, previous=True, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs
    ):
        super(BackwardInputLayer, self).__init__(**kwargs)
        self.layer = layer
        self.slope = slope
        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = mode
            self.convex_domain = convex_domain
        self.finetune = False
        self.previous = previous

    def call_previous(self, inputs):
        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        return w_out_u, b_out_u, w_out_l, b_out_l

    def call_no_previous(self, inputs):

        return get_identity_lirpa(inputs)

    def call(self, inputs):
        if self.previous:
            return self.call_previous(inputs)
        else:
            return self.call_no_previous(inputs)


def get_backward(
    layer, slope=V_slope.name, previous=True, mode=F_HYBRID.name, convex_domain={}, finetune=False, **kwargs
):

    # do it better
    # either a Decomon layer or its pure Keras version
    class_name = layer.__class__.__name__
    if class_name[:7] == "Decomon":
        class_name = "".join(layer.__class__.__name__.split("Decomon")[1:])

    backward_class_name = "Backward{}".format(class_name)
    class_ = globals()[backward_class_name]
    try:
        return class_(
            layer, slope=slope, previous=previous, mode=mode, convex_domain=convex_domain, finetune=finetune, **kwargs
        )
    except KeyError:
        import pdb

        pdb.set_trace()


def join(layer):
    raise NotImplementedError()
