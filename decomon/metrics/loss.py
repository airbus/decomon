from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Input, InputLayer, Lambda, Layer

from decomon.layers.activations import softmax as softmax_
from decomon.layers.utils import F_FORWARD, F_HYBRID, F_IBP, get_lower, get_upper
from decomon.models import DecomonModel
from decomon.models.utils import get_mode
from decomon.utils import set_mode

from ..layers.core import DecomonLayer
from .utils import categorical_cross_entropy


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

    elif mode == F_FORWARD.name:

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

    elif mode == F_HYBRID.name:

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

    else:
        raise ValueError(f"Unknown mode {mode}")

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

        elif mode == F_FORWARD.name:
            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_u = y_pred[:, :-1, n_comp : n_comp + n_out]
            b_u = y_pred[:, -1, n_comp : n_comp + n_out]

        elif mode == F_HYBRID.name:

            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_u = y_pred[:, :-2, n_comp : n_comp + n_out]
            b_u = y_pred[:, -2, n_comp : n_comp + n_out]
            u_c = y_pred[:, -1, n_comp : n_comp + n_out]

        else:
            raise ValueError(f"Unknown mode {mode}")

        if IBP:
            score_ibp = upper_ibp(u_c, y_true)
        if forward:
            score_forward = upper_forward(x_0, w_u, b_u, y_true)

        if mode == F_IBP.name:
            return K.mean(score_ibp)
        elif mode == F_FORWARD.name:
            return K.mean(score_forward)
        elif mode == F_HYBRID.name:
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

        elif mode == F_FORWARD.name:
            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_l = y_pred[:, :-1, n_comp + n_out :]
            b_l = y_pred[:, -1, n_comp + n_out :]

        elif mode == F_HYBRID.name:

            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_l = y_pred[:, :-2, n_comp + n_out :]
            b_l = y_pred[:, -2, n_comp + n_out :]
            l_c = y_pred[:, -1, n_comp + n_out :]

        else:
            raise ValueError(f"Unknown mode {mode}")

        if IBP:
            score_ibp = lower_ibp(l_c, y_true)
        if forward:
            score_forward = lower_forward(x_0, w_l, b_l, y_true)

        if mode == F_IBP.name:
            return K.mean(score_ibp)
        elif mode == F_FORWARD.name:
            return K.mean(score_forward)
        elif mode == F_HYBRID.name:
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

        elif mode == F_FORWARD.name:
            if len(y_pred.shape) == 3:
                x_0 = K.permute_dimensions(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_u = y_pred[:, :-1, n_comp : n_comp + n_out]
            b_u = y_pred[:, -1, n_comp : n_comp + n_out]
            w_l = y_pred[:, :-1, n_comp + n_out :]
            b_l = y_pred[:, -1, n_comp + n_out :]

            if softmax:
                _, w_u, b_u, w_l, b_l = softmax_(
                    [x_0, w_u, b_u, w_l, b_l], mode=mode, convex_domain=convex_domain, clip=False
                )

        elif mode == F_HYBRID.name:

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

            _, u_c, w_u, b_u, l_c, w_l, b_l = softmax_(
                [x_0, u_c, w_u, b_u, l_c, w_l, b_l], mode=mode, convex_domain=convex_domain, clip=False
            )

        else:
            raise ValueError(f"Unknown mode {mode}")

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
        elif mode == F_FORWARD.name:
            if sigmoid:
                return K.mean(K.sigmoid(score_forward))
            else:
                return K.mean(score_forward)
        elif mode == F_HYBRID.name:
            if sigmoid:
                return K.mean(K.sigmoid(K.minimum(score_ibp, score_forward)))
            else:
                return K.mean(K.minimum(score_ibp, score_forward))

        raise NotImplementedError()

    return loss_adv


# create a layer
class DecomonLossFusion(DecomonLayer):
    def __init__(self, mode=F_HYBRID.name, asymptotic=False, backward=False, **kwargs):
        super().__init__(mode=mode, **kwargs)
        self.final_mode = F_IBP.name
        self.asymptotic = asymptotic
        self.backward = backward

    def call_no_backward(self, inputs, **kwargs):

        if not self.asymptotic:

            u_c, l_c = set_mode(inputs, self.final_mode, self.mode, self.convex_domain)

            return -l_c + K.log(K.sum(K.exp(u_c - K.max(u_c, -1)[:, None]), -1))[:, None] + K.max(u_c, -1)[:, None]

        else:
            u_c, l_c = set_mode(inputs, self.final_mode, self.mode, self.convex_domain)  # (None, n_out), (None, n_out)
            shape = u_c.shape[-1]

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

            source_tensor = tf.linalg.diag(K.ones_like(l_c))

            score = K.concatenate([adv_ibp(u_c, l_c, source_tensor[:, i])[:, None] for i in range(shape)], -1)
            return K.maximum(score, -1)  # + 1e-3*K.maximum(K.max(K.abs(u_c), -1)[:,None], K.abs(l_c))

    def call_backward(self, inputs, **kwargs):

        if not self.asymptotic:

            u_c, l_c = set_mode(inputs, self.final_mode, self.mode, self.convex_domain)
            return K.softmax(u_c)

        else:

            raise NotImplementedError()

    def call(self, inputs, **kwargs):
        if self.backward:
            return self.call_backward(inputs, **kwargs)
        else:
            return self.call_no_backward(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[-1]


# new layer for new loss functions
class DecomonRadiusRobust(DecomonLayer):
    def __init__(self, mode=F_HYBRID.name, backward=False, **kwargs):
        super().__init__(mode=mode, **kwargs)

        if self.mode == F_IBP.name:
            raise NotImplementedError

        if len(self.convex_domain):
            raise NotImplementedError()

        self.backward = backward

    def call_no_backward(self, inputs, **kwargs):

        if self.mode == F_HYBRID.name:
            x, _, w_u, b_u, _, w_l, b_l = inputs
        else:
            x, w_u, b_u, w_l, b_l = inputs

        # compute center
        x_0 = (x[:, 0] + x[:, 1]) / 2.0
        radius = K.maximum((x[:, 1] - x[:, 0]) / 2.0, K.epsilon())
        source_tensor = tf.linalg.diag(K.ones_like(b_l))

        shape = b_l.shape[-1]

        def radius_label(y_tensor, backward=False):

            t_tensor = 1 - y_tensor
            s_tensor = y_tensor

            W_adv = (
                K.sum(-w_l * (s_tensor[:, None]), -1, keepdims=True) + w_u * t_tensor[:, None] + w_l * y_tensor[:, None]
            )  # (None, n_in, n_out)
            b_adv = (
                K.sum(-b_l * (s_tensor), -1, keepdims=True) + b_u * t_tensor + (b_l - 1e6) * y_tensor
            )  # (None, n_out)

            score = K.sum(W_adv * x_0[:, :, None], 1) + b_adv  # (None, n_out)

            denum = K.maximum(K.sum(K.abs(W_adv * radius[:, :, None]), 1), K.epsilon())  # (None, n_out)

            eps_adv = K.minimum(-score / denum + y_tensor, 2.0)

            adv_volume = 1.0 - eps_adv

            return K.max(adv_volume, -1, keepdims=True)

        return K.concatenate([radius_label(source_tensor[:, i]) for i in range(shape)], -1)

    def call_backward(self, inputs, **kwargs):

        if self.mode == F_HYBRID.name:
            x, _, w_u, b_u, _, w_l, b_l = inputs
        else:
            x, w_u, b_u, w_l, b_l = inputs

        # compute center
        x_0 = (x[:, 0] + x[:, 1]) / 2.0
        radius = K.maximum((x[:, 1] - x[:, 0]) / 2.0, K.epsilon())
        source_tensor = tf.linalg.diag(K.ones_like(b_l))

        shape = b_l.shape[-1]

        def radius_label(y_tensor):

            t_tensor = 1 - y_tensor
            s_tensor = y_tensor

            W_adv = w_u
            b_adv = b_u - 1e6 * y_tensor

            score = K.sum(W_adv * x_0[:, :, None], 1) + b_adv  # (None, n_out)
            denum = K.maximum(K.sum(K.abs(W_adv * radius[:, :, None]), 1), K.epsilon())  # (None, n_out)

            eps_adv = K.minimum(-score / denum + y_tensor, 2.0)

            adv_volume = 1.0 - eps_adv

            return K.max(adv_volume, -1, keepdims=True)

        return K.concatenate([radius_label(source_tensor[:, i]) for i in range(shape)], -1)

    def call(self, inputs, **kwargs):
        if self.backward:
            return self.call_backward(inputs, **kwargs)
        else:
            return self.call_no_backward(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[-1]


def build_radius_robust_model(model):
    IBP = model.IBP
    forward = model.forward

    mode = get_mode(IBP, forward)
    convex_domain = model.convex_domain

    inputs = model.input
    output = model.output

    layer_robust = DecomonRadiusRobust(mode=mode, convex_domain=convex_domain, backward=model.backward_bounds)
    output_robust = layer_robust(output)

    return DecomonModel(
        input=inputs,
        output=output_robust,
        convex_domain=model.convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=model.finetune,
    )


##### DESIGN LOSS FUNCTIONS
def build_crossentropy_model(model):
    IBP = model.IBP
    forward = model.forward

    mode = get_mode(IBP, forward)
    convex_domain = model.convex_domain

    inputs = model.input
    output = model.output

    layer_fusion = DecomonLossFusion(mode=mode, backward=model.backward_bounds)
    output_fusion = layer_fusion(output, mode=mode)

    return DecomonModel(
        input=inputs,
        output=output_fusion,
        convex_domain=model.convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=model.finetune,
    )


##### DESIGN LOSS FUNCTIONS
def build_asymptotic_crossentropy_model(model):
    IBP = model.IBP
    forward = model.forward

    mode = get_mode(IBP, forward)
    convex_domain = model.convex_domain

    inputs = model.input
    output = model.output

    layer_fusion = DecomonLossFusion(mode=mode, asymptotic=True)
    output_fusion = layer_fusion(output, mode=mode)

    return DecomonModel(
        input=inputs,
        output=output_fusion,
        convex_domain=model.convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=model.finetune,
    )
