from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model

from decomon.utils import get_upper


class MetricMode(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class MetricLayer(ABC, Layer):
    def __init__(
        self,
        ibp: bool,
        affine: bool,
        mode: Union[str, MetricMode],
        convex_domain: Optional[Dict[str, Any]],
        **kwargs: Any,
    ):
        """
        Args:
            ibp: boolean that indicates whether we propagate constant
                bounds
            affine: boolean that indicates whether we propagate affine
                bounds
            mode: str: 'backward' or 'forward' whether we doforward or
                backward linear relaxation
            convex_domain: the type of input convex domain for the
                linear relaxation
            **kwargs
        """
        super().__init__(**kwargs)
        self.ibp = ibp
        self.affine = affine
        self.mode = MetricMode(mode)
        self.convex_domain = convex_domain

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "ibp": self.ibp,
                "affine": self.affine,
                "mode": self.mode,
                "convex_domain": self.convex_domain,
            }
        )
        return config

    @abstractmethod
    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> tf.Tensor:
        """
        Args:
            inputs

        Returns:

        """
        pass


class AdversarialScore(MetricLayer):
    """Training with symbolic LiRPA bounds for promoting adversarial robustness"""

    def __init__(
        self,
        ibp: bool,
        affine: bool,
        mode: Union[str, MetricMode],
        convex_domain: Optional[Dict[str, Any]],
        **kwargs: Any,
    ):
        """
        Args:
            ibp: boolean that indicates whether we propagate constant
                bounds
            forward: boolean that indicates whether we propagate affine
                bounds
            mode: str: 'backward' or 'forward' whether we doforward or
                backward linear relaxation
            convex_domain: the type of input convex domain for the
                linear relaxation
            **kwargs
        """
        super().__init__(ibp=ibp, affine=affine, mode=mode, convex_domain=convex_domain, **kwargs)

    def linear_adv(self, z_tensor, y_tensor, w_u, b_u, w_l, b_l):

        t_tensor = 1 - y_tensor
        w_upper = w_u * (1 - y_tensor[:, None]) - K.expand_dims(K.sum(w_l * y_tensor[:, None], -1), -1)
        b_upper = b_u * (1 - y_tensor) - b_l * y_tensor

        adv_score = get_upper(z_tensor, w_upper, b_upper) - 1e6 * y_tensor

        return K.max(adv_score, -1)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> tf.Tensor:
        """
        Args:
            inputs

        Returns:
            adv_score <0 if the predictionis robust on the input convex
            domain
        """

        y_tensor = inputs[-1]

        def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1.0 - source_tensor

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

        if self.ibp and self.affine:
            _, z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = inputs[:8]
        elif not self.ibp and self.affine:
            _, z, w_u_f, b_u_f, w_l_f, b_l_f = inputs[:6]
        elif self.ibp and not self.affine:
            _, z, u_c, l_c = inputs[:4]
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if self.ibp:
            adv_ibp = get_ibp_score(u_c, l_c, y_tensor)
        if self.affine:
            adv_f = get_forward_score(z, w_u_f, b_u_f, w_l_f, b_l_f, y_tensor)

        if self.ibp and not self.affine:
            adv_score = adv_ibp
        elif self.ibp and self.affine:
            adv_score = K.minimum(adv_ibp, adv_f)
        elif not self.ibp and self.affine:
            adv_score = adv_f
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if self.mode == MetricMode.BACKWARD:
            w_u_b, b_u_b, w_l_b, b_l_b, _ = inputs[-5:]
            adv_b = get_backward_score(z, w_u_b, b_u_b, w_l_b, b_l_b, y_tensor)
            adv_score = K.minimum(adv_score, adv_b)

        return adv_score


class AdversarialCheck(MetricLayer):
    """Training with symbolic LiRPA bounds for promoting adversarial robustness"""

    def __init__(
        self,
        ibp: bool,
        affine: bool,
        mode: Union[str, MetricMode],
        convex_domain: Optional[Dict[str, Any]],
        **kwargs: Any,
    ):
        """
        Args:
            ibp: boolean that indicates whether we propagate constant
                bounds
            forward: boolean that indicates whether we propagate affine
                bounds
            mode: str: 'backward' or 'forward' whether we doforward or
                backward linear relaxation
            convex_domain: the type of input convex domain for the
                linear relaxation
            **kwargs
        """
        super().__init__(ibp=ibp, affine=affine, mode=mode, convex_domain=convex_domain, **kwargs)

    def linear_adv(self, z_tensor, y_tensor, w_u, b_u, w_l, b_l):
        w_upper = w_u * (1 - y_tensor[:, None]) - K.expand_dims(K.sum(w_l * y_tensor[:, None], -1), -1)
        b_upper = b_u * (1 - y_tensor) - b_l * y_tensor

        adv_score = get_upper(z_tensor, w_upper, b_upper) - 1e6 * y_tensor

        return K.max(adv_score, -1)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> tf.Tensor:
        """
        Args:
            inputs

        Returns:
            adv_score <0 if the predictionis robust on the input convex
            domain
        """

        y_tensor = inputs[-1]

        def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1.0 - source_tensor

            shape = np.prod(u_c.shape[1:])
            u_c_ = K.reshape(u_c, (-1, shape))
            l_c_ = K.reshape(l_c, (-1, shape))

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

        if self.ibp and self.affine:
            _, z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = inputs[:8]
        elif not self.ibp and self.affine:
            _, z, w_u_f, b_u_f, w_l_f, b_l_f = inputs[:6]
        elif self.ibp and not self.affine:
            _, z, u_c, l_c = inputs[:4]
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if self.ibp:
            adv_ibp = get_ibp_score(u_c, l_c, y_tensor)
        if self.affine:
            adv_f = get_forward_score(z, w_u_f, b_u_f, w_l_f, b_l_f, y_tensor)

        if self.ibp and not self.affine:
            adv_score = adv_ibp
        elif self.ibp and self.affine:
            adv_score = K.minimum(adv_ibp, adv_f)
        elif not self.ibp and self.affine:
            adv_score = adv_f
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if self.mode == MetricMode.BACKWARD:
            w_u_b, b_u_b, w_l_b, b_l_b, _ = inputs[-5:]
            adv_b = get_backward_score(z, w_u_b, b_u_b, w_l_b, b_l_b, y_tensor)
            adv_score = K.minimum(adv_score, adv_b)

        return adv_score


def build_formal_adv_check_model(decomon_model):
    """automatic design on a Keras  model which predicts a certificate of adversarial robustness

    Args:
        decomon_model

    Returns:

    """
    # check type and that backward pass is available

    convex_domain = decomon_model.convex_domain
    layer = AdversarialCheck(decomon_model.IBP, decomon_model.forward, decomon_model.mode, convex_domain)
    output = decomon_model.output
    input = decomon_model.input
    n_out = decomon_model.output[0].shape[1:]
    y_out = Input(n_out)

    adv_score = layer(output + [y_out])
    adv_model = Model(input + [y_out], adv_score)
    return adv_model


def build_formal_adv_model(decomon_model):
    """automatic design on a Keras  model which predicts a certificate of adversarial robustness

    Args:
        decomon_model

    Returns:

    """
    # check type and that backward pass is available

    convex_domain = decomon_model.convex_domain
    layer = AdversarialScore(decomon_model.IBP, decomon_model.forward, decomon_model.mode, convex_domain)
    output = decomon_model.output
    input = decomon_model.input
    n_out = decomon_model.output[0].shape[1:]
    y_out = Input(n_out)

    adv_score = layer(output + [y_out])
    adv_model = Model(input + [y_out], adv_score)
    return adv_model


class UpperScore(MetricLayer):
    """Training with symbolic LiRPA bounds for limiting the local maximum of a neural network"""

    def __init__(
        self,
        ibp: bool,
        affine: bool,
        mode: Union[str, MetricMode],
        convex_domain: Optional[Dict[str, Any]],
        **kwargs: Any,
    ):
        """
        Args:
            ibp: boolean that indicates whether we propagate constant
                bounds
            forward: boolean that indicates whether we propagate affine
                bounds
            mode: str: 'backward' or 'forward' whether we doforward or
                backward linear relaxation
            convex_domain: the type of input convex domain for the
                linear relaxation
            **kwargs
        """
        super().__init__(ibp=ibp, affine=affine, mode=mode, convex_domain=convex_domain, **kwargs)

    def linear_upper(self, z_tensor, y_tensor, w_u, b_u):
        w_upper = w_u * y_tensor[:, None]
        b_upper = b_u * y_tensor

        upper_score = get_upper(z_tensor, w_upper, b_upper)

        return K.sum(upper_score, -1)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> tf.Tensor:
        """
        Args:
            inputs

        Returns:
            upper_score <=0 if the maximum of the neural network is
            lower than the target
        """

        y_tensor = inputs[-1]
        z_tensor = inputs[1]

        if self.ibp and self.affine:
            _, _, u_c, w_u_f, b_u_f, _, _, _ = inputs[:8]

            upper_ibp = K.sum(u_c * y_tensor, -1)
            upper_forward = self.linear_upper(z_tensor, y_tensor, w_u_f, b_u_f)
            upper_score = K.minimum(upper_ibp, upper_forward)
        elif not self.ibp and self.affine:
            _, _, w_u_f, b_u_f = inputs[:6]
            upper_score = self.linear_upper(z_tensor, y_tensor, w_u_f, b_u_f)
        elif self.ibp and not self.affine:
            _, _, u_c, l_c = inputs[:4]
            upper_score = K.sum(u_c * y_tensor, -1)
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if self.mode == MetricMode.BACKWARD:
            w_u_b, b_u_b, _, _, _ = inputs[-5:]
            upper_backward = self.linear_upper(z_tensor, y_tensor, w_u_b[:, 0], b_u_b[:, 0])
            upper_score = K.minimum(upper_score, upper_backward)

        return upper_score


def build_formal_upper_model(decomon_model):
    """automatic design on a Keras  model which predicts a certificate on the local upper bound

    Args:
        decomon_model

    Returns:

    """
    # check type and that backward pass is available

    convex_domain = decomon_model.convex_domain
    layer = UpperScore(decomon_model.IBP, decomon_model.forward, decomon_model.mode, convex_domain)
    output = decomon_model.output
    input = decomon_model.input
    n_out = decomon_model.output[0].shape[1:]
    y_out = Input(n_out)

    upper_score = layer(output + [y_out])
    upper_model = Model(input + [y_out], upper_score)
    return upper_model
