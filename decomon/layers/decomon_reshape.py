from __future__ import absolute_import
from .core import DecomonLayer
import tensorflow.keras.backend as K

# from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.keras.layers import Reshape, Permute, InputSpec

from .core import F_FORWARD, F_IBP, F_HYBRID


class DecomonReshape(Reshape, DecomonLayer):
    """
    Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    def __init__(self, target_shape, mode=F_HYBRID.name, **kwargs):
        """

        :param data_format:
        :param kwargs:

        """
        super(DecomonReshape, self).__init__(target_shape=target_shape, mode=mode, **kwargs)

        if self.mode == F_HYBRID.name:
            self.input_spec = [
                # InputSpec(min_ndim=1),  # y
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
                # InputSpec(min_ndim=1),  # y
                # InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # l_c
            ]
        if self.mode == F_FORWARD.name:
            self.input_spec = [
                # InputSpec(min_ndim=1),  # y
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

        y_input_shape = input_shape[0]
        super(DecomonReshape, self).build(y_input_shape)

    def call(self, inputs):

        op = super(DecomonReshape, self).call
        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = op(h)
            g_ = op(g)
            nb_tensors -= 2

        if self.mode == F_HYBRID.name:
            # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        if self.mode == F_IBP.name:
            # y, x_0, u_c, l_c = inputs[:4]
            u_c, l_c = inputs[:nb_tensors]
        if self.mode == F_FORWARD.name:
            # y, x_0, w_u, b_u, w_l, b_l = inputs[:6]
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]

        # y_ = op(y)
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
            # output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == F_FORWARD.name:
            # output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
            output = [x_0, w_u_, b_u_, w_l_, b_l_]
        if self.mode == F_IBP.name:
            # output = [y_, x_0, u_c_, l_c_]
            output = [u_c_, l_c_]

        if self.dc_decomp:
            output += [h_, g_]

        return output


class DecomonPermute(Permute, DecomonLayer):
    """
    Forward LiRPA implementation of Reshape layers.
    See Keras official documentation for further details on the Reshape operator
    """

    def __init__(self, dims, mode=F_HYBRID.name, **kwargs):
        """

        :param data_format:
        :param kwargs:

        """
        super(DecomonPermute, self).__init__(dims=dims, mode=mode, **kwargs)

        if self.mode == F_HYBRID.name:
            self.input_spec = [
                # InputSpec(min_ndim=1),  # y
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
                # InputSpec(min_ndim=1),  # y
                # InputSpec(min_ndim=1),  # z
                InputSpec(min_ndim=1),  # u_c
                InputSpec(min_ndim=1),  # l_c
            ]
        if self.mode == F_FORWARD.name:
            self.input_spec = [
                # InputSpec(min_ndim=1),  # y
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

        y_input_shape = input_shape[-1]
        super(DecomonPermute, self).build(y_input_shape)

    def call(self, inputs):

        op = super(DecomonPermute, self).call
        nb_tensors = self.nb_tensors
        if self.dc_decomp:
            h, g = inputs[-2:]
            h_ = op(h)
            g_ = op(g)
            nb_tensors -= 2

        if self.mode == F_HYBRID.name:
            # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:8]
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs[:nb_tensors]
        if self.mode == F_IBP.name:
            # y, x_0, u_c, l_c = inputs[:4]
            u_c, l_c = inputs[:nb_tensors]
        if self.mode == F_FORWARD.name:
            # y, x_0, w_u, b_u, w_l, b_l = inputs[:6]
            x_0, w_u, b_u, w_l, b_l = inputs[:nb_tensors]

        # y_ = op(y)
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
            # output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
            output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        if self.mode == F_FORWARD.name:
            # output = [y_, x_0, w_u_, b_u_, w_l_, b_l_]
            output = [x_0, w_u_, b_u_, w_l_, b_l_]
        if self.mode == F_IBP.name:
            # output = [y_, x_0, u_c_, l_c_]
            output = [u_c_, l_c_]

        if self.dc_decomp:
            output += [h_, g_]

        return output
