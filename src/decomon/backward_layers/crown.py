# extra layers necessary for backward LiRPA
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Layer

from decomon.backward_layers.utils import merge_with_previous
from decomon.layers.core import ForwardMode
from decomon.utils import get_lower, get_upper


class Fuse(Layer):
    def __init__(self, mode, **kwargs):
        super(Fuse, self).__init__(**kwargs)
        self.mode = ForwardMode(mode)

    def call(self, inputs_):
        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]

        if self.mode == ForwardMode.AFFINE:
            x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs
        elif self.mode == ForwardMode.HYBRID:
            x_0, u_c, w_f_u, b_f_u, l_c, w_f_l, b_f_l = inputs
        else:
            return backward_bounds

        return merge_with_previous([w_f_u, b_f_u, w_f_l, b_f_l] + backward_bounds)

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode})
        return config


class Convert2BackwardMode(Layer):
    def __init__(self, mode, convex_domain, **kwargs):
        super().__init__(**kwargs)
        self.mode = ForwardMode(mode)
        self.convex_domain = convex_domain

    def call(self, inputs_):
        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]
        w_out_u, b_out_u, w_out_l, b_out_l = backward_bounds

        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            x_0 = inputs[0]
        else:
            u_c, l_c = inputs
            x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)

        if self.mode == ForwardMode.AFFINE:
            return [x_0] + backward_bounds

        if self.mode == ForwardMode.IBP:
            u_c_ = get_upper(x_0, w_out_u, b_out_u, convex_domain={})
            l_c_ = get_lower(x_0, w_out_l, b_out_l, convex_domain={})
            return [u_c_, l_c_]

        if self.mode == ForwardMode.HYBRID:
            u_c_ = get_upper(x_0, w_out_u, b_out_u, convex_domain=self.convex_domain)
            l_c_ = get_lower(x_0, w_out_l, b_out_l, convex_domain=self.convex_domain)
            return [x_0, u_c_, w_out_u, b_out_u, l_c_, w_out_l, b_out_l]

    def get_config(self):
        config = super().get_config()
        config.update({"mode": self.mode, "convex_domain": self.convex_domain})
        return config


class MergeWithPrevious(Layer):
    def __init__(self, input_shape_layer=None, backward_shape_layer=None, **kwargs):
        super(MergeWithPrevious, self).__init__(**kwargs)
        self.input_shape_layer = input_shape_layer
        self.backward_shape_layer = backward_shape_layer
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape_layer": self.input_shape_layer,
                "backward_shape_layer": self.backward_shape_layer,
            }
        )
        return config


class Convert2Mode(Layer):
    def __init__(self, mode_from, mode_to, convex_domain, **kwargs):
        super(Convert2Mode, self).__init__(**kwargs)
        self.mode_from = ForwardMode(mode_from)
        self.mode_to = ForwardMode(mode_to)
        self.convex_domain = convex_domain

    def call(self, inputs_):

        mode_from = self.mode_from
        mode_to = self.mode_to
        convex_domain = self.convex_domain

        if mode_from == mode_to:
            return inputs_

        if mode_from in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            x_0 = inputs_[0]
        else:
            u_c, l_c = inputs_
            if mode_to in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
                z_value = K.cast(0.0, u_c.dtype)
                o_value = K.cast(1.0, u_c.dtype)
                w = tf.linalg.diag(z_value * l_c)
                w_u = w
                b_u = u_c
                w_l = w
                b_l = l_c

        if mode_from == ForwardMode.AFFINE:
            _, w_u, b_u, w_l, b_l = inputs_
            if mode_to in [ForwardMode.IBP, ForwardMode.HYBRID]:
                u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
                l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
        if mode_from == ForwardMode.IBP:
            u_c, l_c = inputs_
        if mode_from == ForwardMode.HYBRID:
            _, u_c, w_u, b_u, l_c, w_l, b_l = inputs_

        if mode_to == ForwardMode.IBP:
            return [u_c, l_c]
        if mode_to == ForwardMode.AFFINE:
            return [x_0, w_u, b_u, w_l, b_l]
        if mode_to == ForwardMode.HYBRID:
            return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]

    def get_config(self):
        config = super().get_config()
        config.update({"mode_from": self.mode_from, "mode_to": self.mode_to, "convex_domain": self.convex_domain})
        return config
