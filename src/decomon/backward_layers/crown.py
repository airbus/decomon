# extra layers necessary for backward LiRPA
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Layer

from ..backward_layers.utils import merge_with_previous
from ..utils import F_FORWARD, F_HYBRID, F_IBP, get_lower, get_upper


class Fuse(Layer):
    def __init__(self, mode, **kwargs):
        super(Fuse, self).__init__(**kwargs)
        self.mode = mode

    def call(self, inputs_):
        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]

        if self.mode == F_FORWARD.name:
            x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs
        elif self.mode == F_HYBRID.name:
            x_0, u_c, w_f_u, b_f_u, l_c, w_f_l, b_f_l = inputs
        else:
            return backward_bounds

        return merge_with_previous([w_f_u, b_f_u, w_f_l, b_f_l] + backward_bounds)

    def get_config(self):

        config = super(Fuse, self).get_config()
        config.update({"mode": self.mode})
        return config


class Convert_2_backward_mode(Layer):
    def __init__(self, mode, convex_domain, **kwargs):
        super(Convert_2_backward_mode, self).__init__(**kwargs)
        self.mode = mode
        self.convex_domain = convex_domain

    def call(self, inputs_):
        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]
        w_out_u, b_out_u, w_out_l, b_out_l = backward_bounds

        if self.mode in [F_FORWARD.name, F_HYBRID.name]:
            x_0 = inputs[0]
        else:
            u_c, l_c = inputs
            x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)

        if self.mode == F_FORWARD.name:
            return [x_0] + backward_bounds

        if self.mode == F_IBP.name:
            u_c_ = get_upper(x_0, w_out_u, b_out_u, convex_domain={})
            l_c_ = get_lower(x_0, w_out_l, b_out_l, convex_domain={})
            return [u_c_, l_c_]

        if self.mode == F_HYBRID.name:
            u_c_ = get_upper(x_0, w_out_u, b_out_u, convex_domain=self.convex_domain)
            l_c_ = get_lower(x_0, w_out_l, b_out_l, convex_domain=self.convex_domain)
            return [x_0, u_c_, w_out_u, b_out_u, l_c_, w_out_l, b_out_l]

    def get_config(self):
        config = super(Convert_2_backward_mode, self).get_config()
        config.update({"mode": self.mode, "convex_domain": self.convex_domain})
        return config


class MergeWithPrevious(Layer):
    def __init__(self, input_shape_layer=None, backward_shape_layer=None, **kwargs):
        super(MergeWithPrevious, self).__init__(**kwargs)

        if not (input_shape_layer is None) and not (backward_shape_layer is None):
            _, n_in, n_h = input_shape_layer
            _, n_h, n_out = backward_shape_layer

            w_out_spec = InputSpec(ndim=3, axes={-1: n_h, -2: n_in})
            b_out_spec = InputSpec(ndim=2, axes={-1: n_h})
            w_b_spec = InputSpec(ndim=3, axes={-1: n_out, -2: n_h})
            b_b_spec = InputSpec(ndim=2, axes={-1: n_out})
            self.input_spec = [w_out_spec, b_out_spec] * 2 + [w_b_spec, b_b_spec] * 2  #

    def call(self, inputs):
        return merge_with_previous(inputs)


class Convert_2_mode(Layer):
    def __init__(self, mode_from, mode_to, convex_domain, **kwargs):
        super(Convert_2_mode, self).__init__(**kwargs)
        self.mode_from = mode_from
        self.mode_to = mode_to
        self.convex_domain = convex_domain

    def call(self, inputs_):

        mode_from = self.mode_from
        mode_to = self.mode_to
        convex_domain = self.convex_domain

        if mode_from == mode_to:
            return inputs_

        if mode_from in [F_FORWARD.name, F_HYBRID.name]:
            x_0 = inputs_[0]
        else:
            u_c, l_c = inputs_
            if mode_to in [F_FORWARD.name, F_HYBRID.name]:
                x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
                z_value = K.cast(0.0, u_c.dtype)
                o_value = K.cast(1.0, u_c.dtype)
                w = tf.linalg.diag(z_value * l_c)
                w_u = w
                b_u = u_c
                w_l = w
                b_l = l_c

        if mode_from == F_FORWARD.name:
            _, w_u, b_u, w_l, b_l = inputs_
            if mode_to in [F_IBP.name, F_HYBRID.name]:
                u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
                l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
        if mode_from == F_IBP.name:
            u_c, l_c = inputs_
        if mode_from == F_HYBRID.name:
            _, u_c, w_u, b_u, l_c, w_l, b_l = inputs_

        if mode_to == F_IBP.name:
            return [u_c, l_c]
        if mode_to == F_FORWARD.name:
            return [x_0, w_u, b_u, w_l, b_l]
        if mode_to == F_HYBRID.name:
            return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]

    def get_config(self):

        config = super(Convert_2_mode, self).get_config()
        config.update({"mode_from": self.mode_from, "mode_to": self.mode_to, "convex_domain": self.convex_domain})
        return config
