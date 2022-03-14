from __future__ import absolute_import
from tensorflow.keras.layers import Layer
from ..layers.utils import get_upper
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import numpy as np


class Adversarial_score(Layer):
    """
    Training with symbolic LiRPA bounds for promoting adversarial robustness
    """

    def __init__(self, ibp, forward, mode, convex_domain, **kwargs):
        """

        :param ibp: boolean that indicates whether we propagate constant bounds
        :param forward: boolean that indicates whether we propagate affine bounds
        :param mode: str: 'backward' or 'forward' whether we doforward or backward linear relaxation
        :param convex_domain: the type of input convex domain for the linear relaxation
        :param kwargs:
        """
        super(Adversarial_score, self).__init__(**kwargs)
        self.ibp = ibp
        self.forward = forward
        self.mode = mode
        self.convex_domain = convex_domain

    def linear_adv(self, z_tensor, y_tensor, w_u, b_u, w_l, b_l):

        t_tensor = 1 - y_tensor
        mask = K.expand_dims(t_tensor, -1) * K.expand_dims(y_tensor)
        w_upper = w_u * (1 - y_tensor[:, None]) - K.expand_dims(K.sum(w_l * y_tensor[:, None], -1), -1)
        b_upper = b_u * (1 - y_tensor) - b_l * y_tensor

        adv_score = get_upper(z_tensor, w_upper, b_upper) - 1e6 * y_tensor

        return K.max(adv_score, -1)

    def call(self, inputs):
        """

        :param inputs:
        :return: adv_score <0 if the predictionis robust on the input convex domain
        """

        y_tensor = inputs[-1]
        z_tensor = inputs[1]

        def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1.0 - source_tensor

            shape = np.prod(u_c.shape[1:])
            u_c_ = K.reshape(u_c, (-1, shape))
            l_c_ = K.reshape(l_c, (-1, shape))

            # to improve
            score_u = (
                u_c * target_tensor - K.expand_dims(K.min(l_c * source_tensor, -1), -1) - 1e6 * (1 - target_tensor)
            )

            return K.max(score_u, -1)

        def get_forward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1.0 - source_tensor

            n_dim = w_u.shape[1]
            shape = np.prod(b_u.shape[1:])
            w_u_ = K.reshape(w_u, (-1, n_dim, shape, 1))
            w_l_ = K.reshape(w_l, (-1, n_dim, 1, shape))
            b_u_ = K.reshape(b_u, (-1, shape, 1))
            b_l_ = K.reshape(b_l, (-1, 1, shape))

            w_u_f = w_u_ - w_l_
            b_u_f = b_u_ - b_l_

            # add penalties on biases
            b_u_f = b_u_f - 1e6 * (1 - source_tensor)[:, None, :]
            b_u_f = b_u_f - 1e6 * (1 - target_tensor)[:, :, None]

            upper = get_upper(z_tensor, w_u_f, b_u_f)
            return K.max(upper, (-1, -2))

        def get_backward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

            return get_forward_score(z_tensor, w_u[:, 0], b_u[:, 0], w_l[:, 0], b_l[:, 0], source_tensor, target_tensor)

        if self.ibp and self.forward:
            _, z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = inputs[:8]
        if not self.ibp and self.forward:
            _, z, w_u_f, b_u_f, w_l_f, b_l_f = inputs[:6]
        if self.ibp and not self.forward:
            _, z, u_c, l_c = inputs[:4]

        if self.ibp:
            adv_ibp = get_ibp_score(u_c, l_c, y_tensor)
        if self.forward:
            adv_f = get_forward_score(z, w_u_f, b_u_f, w_l_f, b_l_f, y_tensor)

        if self.ibp and not self.forward:
            adv_score = adv_ibp
        if self.ibp and self.forward:
            adv_score = K.minimum(adv_ibp, adv_f)
        if not self.ibp and self.forward:
            adv_score = adv_f

        if self.mode == "backward":
            w_u_b, b_u_b, w_l_b, b_l_b, _ = inputs[-5:]
            adv_b = get_backward_score(z, w_u_b, b_u_b, w_l_b, b_l_b, y_tensor)
            adv_score = K.minimum(adv_score, adv_b)

        return adv_score


class Adversarial_check(Layer):
    """
    Training with symbolic LiRPA bounds for promoting adversarial robustness
    """

    def __init__(self, ibp, forward, mode, convex_domain, **kwargs):
        """

        :param ibp: boolean that indicates whether we propagate constant bounds
        :param forward: boolean that indicates whether we propagate affine bounds
        :param mode: str: 'backward' or 'forward' whether we doforward or backward linear relaxation
        :param convex_domain: the type of input convex domain for the linear relaxation
        :param kwargs:
        """
        super(Adversarial_check, self).__init__(**kwargs)
        self.ibp = ibp
        self.forward = forward
        self.mode = mode
        self.convex_domain = convex_domain

    def linear_adv(self, z_tensor, y_tensor, w_u, b_u, w_l, b_l):
        w_upper = w_u * (1 - y_tensor[:, None]) - K.expand_dims(K.sum(w_l * y_tensor[:, None], -1), -1)
        b_upper = b_u * (1 - y_tensor) - b_l * y_tensor

        adv_score = get_upper(z_tensor, w_upper, b_upper) - 1e6 * y_tensor

        return K.max(adv_score, -1)

    def call(self, inputs):
        """

        :param inputs:
        :return: adv_score <0 if the predictionis robust on the input convex domain
        """

        y_tensor = inputs[-1]
        z_tensor = inputs[1]

        def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1.0 - source_tensor

            shape = np.prod(u_c.shape[1:])
            u_c_ = K.reshape(u_c, (-1, shape))
            l_c_ = K.reshape(l_c, (-1, shape))

            # to improve
            score_u = (
                l_c_ * target_tensor - K.expand_dims(K.min(u_c_ * source_tensor, -1), -1) - 1e6 * (1 - target_tensor)
            )

            return K.max(score_u, -1)

        def get_forward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1.0 - source_tensor

            n_dim = w_u.shape[1]
            shape = np.prod(b_u.shape[1:])
            w_u_ = K.reshape(w_u, (-1, n_dim, shape, 1))
            w_l_ = K.reshape(w_l, (-1, n_dim, 1, shape))
            b_u_ = K.reshape(b_u, (-1, shape, 1))
            b_l_ = K.reshape(b_l, (-1, 1, shape))

            w_u_f = w_l_ - w_u_
            b_u_f = b_l_ - b_u_

            # add penalties on biases
            b_u_f = b_u_f - 1e6 * (1 - target_tensor)[:, None, :]
            b_u_f = b_u_f - 1e6 * (1 - source_tensor)[:, :, None]

            upper = get_upper(z_tensor, w_u_f, b_u_f)
            return K.max(upper, (-1, -2))

        def get_backward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

            return get_forward_score(z_tensor, w_u[:, 0], b_u[:, 0], w_l[:, 0], b_l[:, 0], source_tensor, target_tensor)

        if self.ibp and self.forward:
            _, z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = inputs[:8]
        if not self.ibp and self.forward:
            _, z, w_u_f, b_u_f, w_l_f, b_l_f = inputs[:6]
        if self.ibp and not self.forward:
            _, z, u_c, l_c = inputs[:4]

        if self.ibp:
            adv_ibp = get_ibp_score(u_c, l_c, y_tensor)
        if self.forward:
            adv_f = get_forward_score(z, w_u_f, b_u_f, w_l_f, b_l_f, y_tensor)

        if self.ibp and not self.forward:
            adv_score = adv_ibp
        if self.ibp and self.forward:
            adv_score = K.minimum(adv_ibp, adv_f)
        if not self.ibp and self.forward:
            adv_score = adv_f

        if self.mode == "backward":
            w_u_b, b_u_b, w_l_b, b_l_b, _ = inputs[-5:]
            adv_b = get_backward_score(z, w_u_b, b_u_b, w_l_b, b_l_b, y_tensor)
            adv_score = K.minimum(adv_score, adv_b)

        return adv_score


def build_formal_adv_check_model(decomon_model):
    """
    automatic design on a Keras  model which predicts a certificate of adversarial robustness
    :param decomon_model:
    :return:
    """
    # check type and that backward pass is available

    convex_domain = decomon_model.convex_domain
    layer = Adversarial_check(decomon_model.IBP, decomon_model.forward, decomon_model.mode, convex_domain)
    output = decomon_model.output
    input = decomon_model.input
    n_out = decomon_model.output[0].shape[1:]
    y_out = Input(n_out)

    adv_score = layer(output + [y_out])
    adv_model = Model(input + [y_out], adv_score)
    return adv_model


def build_formal_adv_model(decomon_model):
    """
    automatic design on a Keras  model which predicts a certificate of adversarial robustness
    :param decomon_model:
    :return:
    """
    # check type and that backward pass is available

    convex_domain = decomon_model.convex_domain
    layer = Adversarial_score(decomon_model.IBP, decomon_model.forward, decomon_model.mode, convex_domain)
    output = decomon_model.output
    input = decomon_model.input
    n_out = decomon_model.output[0].shape[1:]
    y_out = Input(n_out)

    adv_score = layer(output + [y_out])
    adv_model = Model(input + [y_out], adv_score)
    return adv_model


class Upper_score(Layer):
    """
    Training with symbolic LiRPA bounds for limiting the local maximum of a neural network
    """

    def __init__(self, ibp, forward, mode, convex_domain, **kwargs):
        super(Upper_score, self).__init__(**kwargs)
        self.ibp = ibp
        self.forward = forward
        self.mode = mode
        self.convex_domain = convex_domain

    def linear_upper(self, z_tensor, y_tensor, w_u, b_u):
        w_upper = w_u * y_tensor[:, None]
        b_upper = b_u * y_tensor

        upper_score = get_upper(z_tensor, w_upper, b_upper)

        return K.sum(upper_score, -1)

    def call(self, inputs):
        """

        :param inputs:
        :return: upper_score <=0 if the maximum of the neural network is lower than the target
        """

        y_tensor = inputs[-1]
        z_tensor = inputs[1]

        if self.ibp and self.forward:
            _, _, u_c, w_u_f, b_u_f, _, _, _ = inputs[:8]

            upper_ibp = K.sum(u_c * y_tensor, -1)
            upper_forward = self.linear_upper(z_tensor, y_tensor, w_u_f, b_u_f)
            upper_score = K.minimum(upper_ibp, upper_forward)

        if not self.ibp and self.forward:
            _, _, w_u_f, b_u_f = inputs[:6]
            upper_score = self.linear_upper(z_tensor, y_tensor, w_u_f)

        if self.ibp and not self.forward:
            _, _, u_c, l_c = inputs[:4]
            upper_score = K.sum(u_c * y_tensor, -1)

        if self.mode == "backward":
            w_u_b, b_u_b, _, _, _ = inputs[-5:]
            upper_backward = self.linear_upper(z_tensor, y_tensor, w_u_b[:, 0], b_u_b[:, 0])
            upper_score = K.minimum(upper_score, upper_backward)

        return upper_score


def build_formal_upper_model(decomon_model):
    """
    automatic design on a Keras  model which predicts a certificate on the local upper bound
    :param decomon_model:
    :return:
    """
    # check type and that backward pass is available

    convex_domain = decomon_model.convex_domain
    layer = Upper_score(decomon_model.IBP, decomon_model.forward, decomon_model.mode, convex_domain)
    output = decomon_model.output
    input = decomon_model.input
    n_out = decomon_model.output[0].shape[1:]
    y_out = Input(n_out)

    upper_score = layer(output + [y_out])
    upper_model = Model(input + [y_out], upper_score)
    return upper_model




