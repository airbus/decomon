from __future__ import absolute_import
import tensorflow.keras.backend as K
from decomon.models.utils import get_mode
from decomon.layers.utils import F_HYBRID, F_FORWARD, F_IBP, get_upper, get_lower
from decomon.layers.activations import softmax as softmax_
from decomon.models import DecomonModel
from tensorflow.keras.layers import InputLayer, Input, Layer, Flatten, Lambda
import numpy as np


def get_model(model):
    IBP = model.IBP
    forward = model.forward

    mode = get_mode(IBP, forward)
    convex_domain = model.convex_domain

    inputs = model.input
    output = model.output

    if mode == F_IBP.name:

        def func(output_):
            u_c, l_c = output_
            return K.concatenate([K.expand_dims(u_c, -1), K.expand_dims(l_c, -1)], -1)

        output_ = Lambda(func)(output)

    if mode == F_FORWARD.name:

        def func(output_):
            x_0, w_u, b_u, w_l, b_l = output_
            if len(x_0.shape) == 2:
                x_0_ = x_0[:, :, None]
            else:
                x_0_ = K.permute_dimensions(x_0, (0, 2, 1))
            x_fake = K.sum(x_0_, 1)[:, None]
            x_0_ = K.concatenate([x_0_, x_fake], 1)  # (None, n_in+1, n_comp)

            w_b_u = K.concatenate([w_u, b_u[:, None]], 1)  # (None, n_in+1, n_out)
            w_b_l = K.concatenate([w_l, b_l[:, None]], 1)

            w_b = K.concatenate([w_b_u, w_b_l], -1)  # (None, n_in+1, 2*n_out)
            return K.concatenate([x_0_, w_b], -1)  # (None, n_in+1, n_comp+2*n_out)

        output_ = Lambda(func)(output)

    if mode == F_HYBRID.name:

        def func(output_):
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = output_

            if len(x_0.shape) == 2:
                x_0_ = x_0[:, :, None]
            else:
                x_0_ = K.permute_dimensions(x_0, (0, 2, 1))
            x_fake = K.sum(x_0_, 1)[:, None]
            x_0_ = K.concatenate([x_0_, x_fake, x_fake], 1)  # (None, n_in+2, n_comp)

            w_b_u = K.concatenate([w_u, b_u[:, None]], 1)  # (None, n_in+1, n_out)
            w_b_l = K.concatenate([w_l, b_l[:, None]], 1)

            w_b = K.concatenate([w_b_u, w_b_l], -1)  # (None, n_in+1, 2*n_out)

            u_l = K.concatenate([u_c, l_c], 1)[:, None]  # (None, 1, 2*n_out)

            w_b_u_l = K.concatenate([w_b, u_l], 1)  # (None, n_in+2, 2*n_out)

            return K.concatenate([x_0_, w_b_u_l], -1)  # (None, n_in+1, n_comp+2*n_out)

        output_ = Lambda(func)(output)

    return DecomonModel(
        input=inputs,
        output=output_,
        convex_domain=model.convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=model.finetune,
    )

def get_upper_loss(model):

    IBP = model.IBP
    forward = model.forward

    mode = get_mode(IBP, forward)
    convex_domain = model.convex_domain

    if forward:
        if len(convex_domain) == 0 or convex_domain["name"] == "ball":
            n_comp = 2
        else:
            n_comp = 1

    n_out = np.prod(model.output[-1].shape[1:])

    def upper_ibp(u_c, u_ref):
        # minimize the upper bound compared to the reference
        return K.max(u_c - u_ref, -1)

    def upper_forward(x, w_u, b_u, u_ref):

        upper = get_upper(x, w_u, b_u, convex_domain=convex_domain)

        return K.max(upper - u_ref, -1)

    def loss_upper(y_true, y_pred):

        if mode == F_IBP.name:
            u_c = y_pred[:, :, 0]

        if mode == F_FORWARD.name:
            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_u = y_pred[:, :-1, n_comp : n_comp + n_out]
            b_u = y_pred[:, -1, n_comp : n_comp + n_out]

        if mode == F_HYBRID.name:

            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_u = y_pred[:, :-2, n_comp : n_comp + n_out]
            b_u = y_pred[:, -2, n_comp : n_comp + n_out]
            u_c = y_pred[:, -1, n_comp : n_comp + n_out]

        if IBP:
            score_ibp = upper_ibp(u_c, y_true)
        if forward:
            score_forward = upper_forward(x_0, w_u, b_u, y_true)

        if mode == F_IBP.name:
            return K.mean(score_ibp)
        if mode == F_FORWARD.name:
            return K.mean(score_forward)
        if mode == F_HYBRID.name:
            return K.mean(K.minimum(score_ibp, score_forward))

        raise NotImplementedError()

    return loss_upper


def get_lower_loss(model):

    IBP = model.IBP
    forward = model.forward

    mode = get_mode(IBP, forward)
    convex_domain = model.convex_domain

    if forward:
        if len(convex_domain) == 0 or convex_domain["name"] == "ball":
            n_comp = 2
        else:
            n_comp = 1

    n_out = np.prod(model.output[-1].shape[1:])

    def lower_ibp(l_c, l_ref):
        # minimize the upper bound compared to the reference
        return K.max(l_ref - l_c, -1)

    def lower_forward(x, w_l, b_l, l_ref):

        lower = get_lower(x, w_l, b_l, convex_domain=convex_domain)

        return K.max(l_ref - lower, -1)

    def loss_lower(y_true, y_pred):

        if mode == F_IBP.name:
            l_c = y_pred[:, :, 1]

        if mode == F_FORWARD.name:
            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_l = y_pred[:, :-1, n_comp + n_out :]
            b_l = y_pred[:, -1, n_comp + n_out :]

        if mode == F_HYBRID.name:

            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_l = y_pred[:, :-2, n_comp + n_out :]
            b_l = y_pred[:, -2, n_comp + n_out :]
            l_c = y_pred[:, -1, n_comp + n_out :]

        if IBP:
            score_ibp = lower_ibp(l_c, y_true)
        if forward:
            score_forward = lower_forward(x_0, w_l, b_l, y_true)

        if mode == F_IBP.name:
            return K.mean(score_ibp)
        if mode == F_FORWARD.name:
            return K.mean(score_forward)
        if mode == F_HYBRID.name:
            return K.mean(K.minimum(score_ibp, score_forward))

        raise NotImplementedError()

    return loss_lower


def get_adv_loss(model, sigmoid=False, clip_value=None, softmax=False):

    IBP = model.IBP
    forward = model.forward

    mode = get_mode(IBP, forward)
    convex_domain = model.convex_domain

    if forward:
        if len(convex_domain) == 0 or convex_domain["name"] == "ball":
            n_comp = 2
        else:
            n_comp = 1

    n_out = np.prod(model.output[-1].shape[1:])

    def adv_ibp(u_c, l_c, y_tensor):

        t_tensor = 1 - y_tensor
        s_tensor = y_tensor

        t_tensor = t_tensor[:, :, None]
        s_tensor = s_tensor[:, None, :]
        M = t_tensor * s_tensor
        upper = K.expand_dims(u_c, -1) - K.expand_dims(l_c, 1)
        const = (K.max(upper, (-1, -2)) - K.min(upper, (-1, -2)))[:, None, None]
        upper = upper - (const + K.cast(1, K.floatx())) * (1 - M)
        return K.max(upper, (-1, -2))

    def adv_forward(x, w_u, b_u, w_l, b_l, y_tensor):

        w_u_ = K.expand_dims(w_u, -1)
        w_l_ = K.expand_dims(w_l, -2)

        w_adv = w_u_ - w_l_
        b_adv = K.expand_dims(b_u, -1) - K.expand_dims(b_l, 1)

        upper = get_upper(x, w_adv, b_adv, convex_domain=convex_domain)

        t_tensor = 1 - y_tensor
        s_tensor = y_tensor

        t_tensor = t_tensor[:, :, None]
        s_tensor = s_tensor[:, None, :]
        M = t_tensor * s_tensor

        const = (K.max(upper, (-1, -2)) - K.min(upper, (-1, -2)))[:, None, None]
        upper = upper - (const + K.cast(1, K.floatx())) * (1 - M)
        return K.max(upper, (-1, -2))

    def loss_adv(y_true, y_pred):

        if mode == F_IBP.name:
            u_c = y_pred[:, :, 0]
            l_c = y_pred[:, :, 1]

            if softmax:
                u_c, l_c = softmax_([u_c, l_c], mode=mode, convex_domain=convex_domain, clip=False)

        if mode == F_FORWARD.name:
            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_u = y_pred[:, :-1, n_comp : n_comp + n_out]
            b_u = y_pred[:, -1, n_comp : n_comp + n_out]
            w_l = y_pred[:, :-1, n_comp + n_out :]
            b_l = y_pred[:, -1, n_comp + n_out :]

            if softmax:
                _, w_u, b_u, w_l, b_l = softmax_([x_0, w_u, b_u, w_l, b_l],
                                                mode=mode, convex_domain=convex_domain, clip=False)

        if mode == F_HYBRID.name:

            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_u = y_pred[:, :-2, n_comp : n_comp + n_out]
            b_u = y_pred[:, -2, n_comp : n_comp + n_out]
            w_l = y_pred[:, :-2, n_comp + n_out :]
            b_l = y_pred[:, -2, n_comp + n_out :]
            u_c = y_pred[:, -1, n_comp : n_comp + n_out]
            l_c = y_pred[:, -1, n_comp + n_out :]

            _, u_c, w_u, b_u, l_c, w_l, b_l = softmax_([x_0, u_c, w_u, b_u, l_c, w_l, b_l],
                                            mode=mode, convex_domain=convex_domain, clip=False)

        if IBP:
            score_ibp = adv_ibp(u_c, l_c, y_true)
            if clip_value is not None:
                score_ibp = K.maximum(score_ibp, clip_value)
        if forward:
            score_forward = adv_forward(x_0, w_u, b_u, w_l, b_l, y_true)
            if clip_value is not None:
                score_forward = K.maximum(score_forward, clip_value)

        if mode == F_IBP.name:
            if sigmoid:
                return K.mean(K.sigmoid(score_ibp))
            else:
                return K.mean(score_ibp)
        if mode == F_FORWARD.name:
            if sigmoid:
                return K.mean(K.sigmoid(score_forward))
            else:
                return K.mean(score_forward)
        if mode == F_HYBRID.name:
            if sigmoid:
                return K.mean(K.sigmoid(K.minimum(score_ibp, score_forward)))
            else:
                return K.mean(K.minimum(score_ibp, score_forward))

        raise NotImplementedError()

    return loss_adv
