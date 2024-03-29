from collections.abc import Callable
from typing import Any, Optional, Union

import keras
import keras.ops as K
import numpy as np
from keras.config import epsilon
from keras.layers import Lambda, Layer

from decomon.core import (
    BallDomain,
    BoxDomain,
    ForwardMode,
    PerturbationDomain,
    get_mode,
)
from decomon.layers.activations import softmax as softmax_
from decomon.layers.core import DecomonLayer
from decomon.models.models import DecomonModel
from decomon.models.utils import Convert2Mode
from decomon.types import BackendTensor, Tensor


def get_model(model: DecomonModel) -> DecomonModel:
    ibp = model.ibp
    affine = model.affine

    mode = get_mode(ibp, affine)
    perturbation_domain = model.perturbation_domain

    inputs = model.inputs
    outputs = model.outputs
    new_output: keras.KerasTensor

    if mode == ForwardMode.IBP:

        def func(outputs: list[Tensor]) -> Tensor:
            u_c, l_c = outputs
            return K.concatenate([K.expand_dims(u_c, -1), K.expand_dims(l_c, -1)], -1)

        new_output = Lambda(func)(outputs)

    elif mode == ForwardMode.AFFINE:

        def func(outputs: list[Tensor]) -> Tensor:
            x_0, w_u, b_u, w_l, b_l = outputs
            if len(x_0.shape) == 2:
                x_0_reshaped = x_0[:, :, None]
            else:
                x_0_reshaped = K.transpose(x_0, (0, 2, 1))
            x_fake = K.sum(x_0_reshaped, 1)[:, None]
            x_0_reshaped = K.concatenate([x_0_reshaped, x_fake], 1)  # (None, n_in+1, n_comp)

            w_b_u = K.concatenate([w_u, b_u[:, None]], 1)  # (None, n_in+1, n_out)
            w_b_l = K.concatenate([w_l, b_l[:, None]], 1)

            w_b = K.concatenate([w_b_u, w_b_l], -1)  # (None, n_in+1, 2*n_out)
            return K.concatenate([x_0_reshaped, w_b], -1)  # (None, n_in+1, n_comp+2*n_out)

        new_output = Lambda(func)(outputs)

    elif mode == ForwardMode.HYBRID:

        def func(outputs: list[Tensor]) -> Tensor:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = outputs

            if len(x_0.shape) == 2:
                x_0_reshaped = x_0[:, :, None]
            else:
                x_0_reshaped = K.transpose(x_0, (0, 2, 1))
            x_fake = K.sum(x_0_reshaped, 1)[:, None]
            x_0_reshaped = K.concatenate([x_0_reshaped, x_fake, x_fake], 1)  # (None, n_in+2, n_comp)

            w_b_u = K.concatenate([w_u, b_u[:, None]], 1)  # (None, n_in+1, n_out)
            w_b_l = K.concatenate([w_l, b_l[:, None]], 1)

            w_b = K.concatenate([w_b_u, w_b_l], -1)  # (None, n_in+1, 2*n_out)

            u_l = K.concatenate([u_c, l_c], 1)[:, None]  # (None, 1, 2*n_out)

            w_b_u_l = K.concatenate([w_b, u_l], 1)  # (None, n_in+2, 2*n_out)

            return K.concatenate([x_0_reshaped, w_b_u_l], -1)  # (None, n_in+1, n_comp+2*n_out)

        new_output = Lambda(func)(outputs)

    else:
        raise ValueError(f"Unknown mode {mode}")

    return DecomonModel(
        inputs=inputs,
        outputs=new_output,
        perturbation_domain=model.perturbation_domain,
        ibp=ibp,
        affine=affine,
        finetune=model.finetune,
    )


def get_upper_loss(model: DecomonModel) -> Callable[[Tensor, Tensor], Tensor]:
    ibp = model.ibp
    affine = model.affine

    mode = get_mode(ibp, affine)
    perturbation_domain = model.perturbation_domain

    n_comp = perturbation_domain.get_nb_x_components()
    n_out = np.prod(model.output[-1].shape[1:])

    def upper_ibp(u_c: Tensor, u_ref: Tensor) -> Tensor:
        # minimize the upper bound compared to the reference
        return K.max(u_c - u_ref, -1)

    def upper_affine(x: Tensor, w_u: Tensor, b_u: Tensor, u_ref: Tensor) -> Tensor:
        upper = perturbation_domain.get_upper(x, w_u, b_u)

        return K.max(upper - u_ref, -1)

    def loss_upper(y_true: Tensor, y_pred: Tensor) -> Tensor:
        if mode == ForwardMode.IBP:
            u_c = y_pred[:, :, 0]

        elif mode == ForwardMode.AFFINE:
            if len(y_pred.shape) == 3:
                x_0 = K.transpose(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_u = y_pred[:, :-1, n_comp : n_comp + n_out]
            b_u = y_pred[:, -1, n_comp : n_comp + n_out]

        elif mode == ForwardMode.HYBRID:
            if len(y_pred.shape) == 3:
                x_0 = K.transpose(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_u = y_pred[:, :-2, n_comp : n_comp + n_out]
            b_u = y_pred[:, -2, n_comp : n_comp + n_out]
            u_c = y_pred[:, -1, n_comp : n_comp + n_out]

        else:
            raise ValueError(f"Unknown mode {mode}")

        if ibp:
            score_ibp = upper_ibp(u_c, y_true)
        if affine:
            score_affine = upper_affine(x_0, w_u, b_u, y_true)

        if mode == ForwardMode.IBP:
            return K.mean(score_ibp)
        elif mode == ForwardMode.AFFINE:
            return K.mean(score_affine)
        elif mode == ForwardMode.HYBRID:
            return K.mean(K.minimum(score_ibp, score_affine))

        raise NotImplementedError()

    return loss_upper


def get_lower_loss(model: DecomonModel) -> Callable[[Tensor, Tensor], Tensor]:
    ibp = model.ibp
    affine = model.affine

    mode = get_mode(ibp, affine)
    perturbation_domain = model.perturbation_domain

    n_comp = perturbation_domain.get_nb_x_components()
    n_out = np.prod(model.output[-1].shape[1:])

    def lower_ibp(l_c: Tensor, l_ref: Tensor) -> Tensor:
        # minimize the upper bound compared to the reference
        return K.max(l_ref - l_c, -1)

    def lower_affine(x: Tensor, w_l: Tensor, b_l: Tensor, l_ref: Tensor) -> Tensor:
        lower = perturbation_domain.get_lower(x, w_l, b_l)

        return K.max(l_ref - lower, -1)

    def loss_lower(y_true: Tensor, y_pred: Tensor) -> Tensor:
        if mode == ForwardMode.IBP:
            l_c = y_pred[:, :, 1]

        elif mode == ForwardMode.AFFINE:
            if len(y_pred.shape) == 3:
                x_0 = K.transpose(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_l = y_pred[:, :-1, n_comp + n_out :]
            b_l = y_pred[:, -1, n_comp + n_out :]

        elif mode == ForwardMode.HYBRID:
            if len(y_pred.shape) == 3:
                x_0 = K.transpose(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_l = y_pred[:, :-2, n_comp + n_out :]
            b_l = y_pred[:, -2, n_comp + n_out :]
            l_c = y_pred[:, -1, n_comp + n_out :]

        else:
            raise ValueError(f"Unknown mode {mode}")

        if ibp:
            score_ibp = lower_ibp(l_c, y_true)
        if affine:
            score_affine = lower_affine(x_0, w_l, b_l, y_true)

        if mode == ForwardMode.IBP:
            return K.mean(score_ibp)
        elif mode == ForwardMode.AFFINE:
            return K.mean(score_affine)
        elif mode == ForwardMode.HYBRID:
            return K.mean(K.minimum(score_ibp, score_affine))

        raise NotImplementedError()

    return loss_lower


def get_adv_loss(
    model: DecomonModel, sigmoid: bool = False, clip_value: Optional[float] = None, softmax: bool = False
) -> Callable[[Tensor, Tensor], Tensor]:
    ibp = model.ibp
    affine = model.affine

    mode = get_mode(ibp, affine)
    perturbation_domain = model.perturbation_domain

    n_comp = perturbation_domain.get_nb_x_components()
    n_out = np.prod(model.output[-1].shape[1:])

    def adv_ibp(u_c: Tensor, l_c: Tensor, y_tensor: Tensor) -> Tensor:
        t_tensor: Tensor = 1 - y_tensor
        s_tensor = y_tensor

        t_tensor = t_tensor[:, :, None]
        s_tensor = s_tensor[:, None, :]
        M = t_tensor * s_tensor
        upper = K.expand_dims(u_c, -1) - K.expand_dims(l_c, 1)
        const = (K.max(upper, (-1, -2)) - K.min(upper, (-1, -2)))[:, None, None]
        upper = upper - (const + K.cast(1, const.dtype)) * (1 - M)
        return K.max(upper, (-1, -2))

    def adv_affine(
        x: Tensor,
        w_u: Tensor,
        b_u: Tensor,
        w_l: Tensor,
        b_l: Tensor,
        y_tensor: Tensor,
    ) -> Tensor:
        w_u_reshaped = K.expand_dims(w_u, -1)
        w_l_reshaped = K.expand_dims(w_l, -2)

        w_adv = w_u_reshaped - w_l_reshaped
        b_adv = K.expand_dims(b_u, -1) - K.expand_dims(b_l, 1)

        upper = perturbation_domain.get_upper(x, w_adv, b_adv)

        t_tensor: Tensor = 1 - y_tensor
        s_tensor = y_tensor

        t_tensor = t_tensor[:, :, None]
        s_tensor = s_tensor[:, None, :]
        M = t_tensor * s_tensor

        const = (K.max(upper, (-1, -2)) - K.min(upper, (-1, -2)))[:, None, None]
        upper = upper - (const + K.cast(1, const.dtype)) * (1 - M)
        return K.max(upper, (-1, -2))

    def loss_adv(y_true: Tensor, y_pred: Tensor) -> Tensor:
        if mode == ForwardMode.IBP:
            u_c = y_pred[:, :, 0]
            l_c = y_pred[:, :, 1]

            if softmax:
                u_c, l_c = softmax_([u_c, l_c], mode=mode, perturbation_domain=perturbation_domain, clip=False)

        elif mode == ForwardMode.AFFINE:
            if len(y_pred.shape) == 3:
                x_0 = K.transpose(y_pred[:, :-1, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-1, 0]

            w_u = y_pred[:, :-1, n_comp : n_comp + n_out]
            b_u = y_pred[:, -1, n_comp : n_comp + n_out]
            w_l = y_pred[:, :-1, n_comp + n_out :]
            b_l = y_pred[:, -1, n_comp + n_out :]

            if softmax:
                _, w_u, b_u, w_l, b_l = softmax_(
                    [x_0, w_u, b_u, w_l, b_l], mode=mode, perturbation_domain=perturbation_domain, clip=False
                )

        elif mode == ForwardMode.HYBRID:
            if len(y_pred.shape) == 3:
                x_0 = K.transpose(y_pred[:, :-2, :n_comp], (0, 2, 1))
            else:
                x_0 = y_pred[:, :-2, 0]

            w_u = y_pred[:, :-2, n_comp : n_comp + n_out]
            b_u = y_pred[:, -2, n_comp : n_comp + n_out]
            w_l = y_pred[:, :-2, n_comp + n_out :]
            b_l = y_pred[:, -2, n_comp + n_out :]
            u_c = y_pred[:, -1, n_comp : n_comp + n_out]
            l_c = y_pred[:, -1, n_comp + n_out :]

            _, u_c, w_u, b_u, l_c, w_l, b_l = softmax_(
                [x_0, u_c, w_u, b_u, l_c, w_l, b_l], mode=mode, perturbation_domain=perturbation_domain, clip=False
            )

        else:
            raise ValueError(f"Unknown mode {mode}")

        if ibp:
            score_ibp = adv_ibp(u_c, l_c, y_true)
            if clip_value is not None:
                score_ibp = K.maximum(score_ibp, K.cast(clip_value, dtype=score_ibp.dtype))
        if affine:
            score_affine = adv_affine(x_0, w_u, b_u, w_l, b_l, y_true)
            if clip_value is not None:
                score_affine = K.maximum(score_affine, K.cast(clip_value, dtype=score_affine.dtype))

        if mode == ForwardMode.IBP:
            if sigmoid:
                return K.mean(K.sigmoid(score_ibp))
            else:
                return K.mean(score_ibp)
        elif mode == ForwardMode.AFFINE:
            if sigmoid:
                return K.mean(K.sigmoid(score_affine))
            else:
                return K.mean(score_affine)
        elif mode == ForwardMode.HYBRID:
            if sigmoid:
                return K.mean(K.sigmoid(K.minimum(score_ibp, score_affine)))
            else:
                return K.mean(K.minimum(score_ibp, score_affine))

        raise NotImplementedError()

    return loss_adv


def _create_identity_tensor_like(x: Tensor) -> BackendTensor:
    identity_tensor = K.identity(x.shape[-1])
    n_repeat = int(np.prod(x.shape[:-1]))
    return K.reshape(K.repeat(identity_tensor[None], n_repeat, axis=0), tuple(x.shape) + (-1,))


# create a layer
class DecomonLossFusion(DecomonLayer):
    original_keras_layer_class = Layer

    def __init__(
        self,
        asymptotic: bool = False,
        backward: bool = False,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        self.convert2mode_layer = Convert2Mode(
            mode_from=mode,
            mode_to=ForwardMode.IBP,
            perturbation_domain=self.perturbation_domain,
        )
        self.asymptotic = asymptotic
        self.backward = backward

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "asymptotic": self.asymptotic,
                "backward": self.backward,
            }
        )
        return config

    def call_no_backward(self, inputs: list[BackendTensor], **kwargs: Any) -> BackendTensor:
        if not self.asymptotic:
            u_c, l_c = self.convert2mode_layer(inputs)

            return -l_c + K.log(K.sum(K.exp(u_c - K.max(u_c, -1)[:, None]), -1))[:, None] + K.max(u_c, -1)[:, None]

        else:
            u_c, l_c = self.convert2mode_layer(inputs)
            shape = u_c.shape[-1]

            def adv_ibp(u_c: BackendTensor, l_c: BackendTensor, y_tensor: BackendTensor) -> BackendTensor:
                t_tensor = 1 - y_tensor
                s_tensor = y_tensor

                t_tensor = t_tensor[:, :, None]
                s_tensor = s_tensor[:, None, :]
                M = t_tensor * s_tensor
                upper = K.expand_dims(u_c, -1) - K.expand_dims(l_c, 1)
                const = (K.max(upper, (-1, -2)) - K.min(upper, (-1, -2)))[:, None, None]
                upper = upper - (const + K.cast(1, const.dtype)) * (1 - M)
                return K.max(upper, (-1, -2))

            source_tensor = _create_identity_tensor_like(l_c)

            score = K.concatenate([adv_ibp(u_c, l_c, source_tensor[:, i])[:, None] for i in range(shape)], -1)
            return K.maximum(
                score, K.cast(-1, dtype=score.dtype)
            )  # + 1e-3*K.maximum(K.max(K.abs(u_c), -1)[:,None], K.abs(l_c))

    def call_backward(self, inputs: list[BackendTensor], **kwargs: Any) -> BackendTensor:
        if not self.asymptotic:
            u_c, l_c = self.convert2mode_layer(inputs)
            return K.softmax(u_c)

        else:
            raise NotImplementedError()

    def call(self, inputs: list[BackendTensor], **kwargs: Any) -> BackendTensor:
        if self.backward:
            return self.call_backward(inputs, **kwargs)
        else:
            return self.call_no_backward(inputs, **kwargs)

    def compute_output_shape(self, input_shape: list[tuple[Optional[int], ...]]) -> tuple[Optional[int], ...]:  # type: ignore
        return input_shape[-1]

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        return None


# new layer for new loss functions
class DecomonRadiusRobust(DecomonLayer):
    original_keras_layer_class = Layer

    def __init__(
        self,
        backward: bool = False,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        if self.mode == ForwardMode.IBP:
            raise NotImplementedError

        if not isinstance(self.perturbation_domain, BoxDomain):
            raise NotImplementedError()

        self.backward = backward

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "backward": self.backward,
            }
        )
        return config

    def call_no_backward(self, inputs: list[BackendTensor], **kwargs: Any) -> BackendTensor:
        if self.mode == ForwardMode.HYBRID:
            x, _, w_u, b_u, _, w_l, b_l = inputs
        else:
            x, w_u, b_u, w_l, b_l = inputs

        # compute center
        x_0 = (x[:, 0] + x[:, 1]) / 2.0
        radius = K.maximum((x[:, 1] - x[:, 0]) / 2.0, K.cast(epsilon(), dtype=x.dtype))
        source_tensor = _create_identity_tensor_like(b_l)

        shape = b_l.shape[-1]

        def radius_label(y_tensor: BackendTensor, backward: bool = False) -> BackendTensor:
            t_tensor = 1 - y_tensor
            s_tensor = y_tensor

            W_adv = (
                K.sum(-w_l * (s_tensor[:, None]), -1, keepdims=True) + w_u * t_tensor[:, None] + w_l * y_tensor[:, None]
            )  # (None, n_in, n_out)
            b_adv = K.sum(-b_l * s_tensor, -1, keepdims=True) + b_u * t_tensor + (b_l - 1e6) * y_tensor  # (None, n_out)

            score = K.sum(W_adv * x_0[:, :, None], 1) + b_adv  # (None, n_out)

            denum = K.maximum(
                K.sum(K.abs(W_adv * radius[:, :, None]), 1), K.cast(epsilon(), dtype=W_adv.dtype)
            )  # (None, n_out)

            eps_adv = K.minimum(-score / denum + y_tensor, K.cast(2.0, dtype=score.dtype))

            adv_volume = 1.0 - eps_adv

            return K.max(adv_volume, -1, keepdims=True)

        return K.concatenate([radius_label(source_tensor[:, i]) for i in range(shape)], -1)

    def call_backward(self, inputs: list[BackendTensor], **kwargs: Any) -> BackendTensor:
        if self.mode == ForwardMode.HYBRID:
            x, _, w_u, b_u, _, w_l, b_l = inputs
        else:
            x, w_u, b_u, w_l, b_l = inputs

        # compute center
        x_0 = (x[:, 0] + x[:, 1]) / 2.0
        radius = K.maximum((x[:, 1] - x[:, 0]) / 2.0, K.cast(epsilon(), dtype=x.dtype))
        source_tensor = _create_identity_tensor_like(b_l)

        shape = b_l.shape[-1]

        def radius_label(y_tensor: BackendTensor) -> BackendTensor:
            W_adv = w_u
            b_adv = b_u - 1e6 * y_tensor

            score = K.sum(W_adv * x_0[:, :, None], 1) + b_adv  # (None, n_out)
            denum = K.maximum(
                K.sum(K.abs(W_adv * radius[:, :, None]), 1), K.cast(epsilon(), dtype=W_adv.dtype)
            )  # (None, n_out)

            eps_adv = K.minimum(-score / denum + y_tensor, K.cast(2.0, dtype=score.dtype))

            adv_volume = 1.0 - eps_adv

            return K.max(adv_volume, -1, keepdims=True)

        return K.concatenate([radius_label(source_tensor[:, i]) for i in range(shape)], -1)

    def call(self, inputs: list[BackendTensor], **kwargs: Any) -> BackendTensor:
        if self.backward:
            return self.call_backward(inputs, **kwargs)
        else:
            return self.call_no_backward(inputs, **kwargs)

    def compute_output_shape(self, input_shape: Union[list[tuple[Optional[int], ...]],]) -> tuple[Optional[int], ...]:  # type: ignore
        return input_shape[-1]

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        return None


def build_radius_robust_model(model: DecomonModel) -> DecomonModel:
    ibp = model.ibp
    affine = model.affine

    mode = get_mode(ibp, affine)
    perturbation_domain = model.perturbation_domain

    inputs = model.input
    output = model.output

    layer_robust = DecomonRadiusRobust(
        mode=mode, perturbation_domain=perturbation_domain, backward=model.backward_bounds
    )
    output_robust = layer_robust(output)

    return DecomonModel(
        inputs=inputs,
        outputs=output_robust,
        perturbation_domain=model.perturbation_domain,
        ibp=ibp,
        affine=affine,
        finetune=model.finetune,
    )


##### DESIGN LOSS FUNCTIONS
def build_crossentropy_model(model: DecomonModel) -> DecomonModel:
    ibp = model.ibp
    affine = model.affine

    mode = get_mode(ibp, affine)
    perturbation_domain = model.perturbation_domain

    inputs = model.input
    output = model.output

    layer_fusion = DecomonLossFusion(mode=mode, backward=model.backward_bounds)
    output_fusion = layer_fusion(output, mode=mode)

    return DecomonModel(
        inputs=inputs,
        outputs=output_fusion,
        perturbation_domain=model.perturbation_domain,
        ibp=ibp,
        affine=affine,
        finetune=model.finetune,
    )


##### DESIGN LOSS FUNCTIONS
def build_asymptotic_crossentropy_model(model: DecomonModel) -> DecomonModel:
    ibp = model.ibp
    affine = model.affine

    mode = get_mode(ibp, affine)
    perturbation_domain = model.perturbation_domain

    inputs = model.input
    output = model.output

    layer_fusion = DecomonLossFusion(mode=mode, asymptotic=True)
    output_fusion = layer_fusion(output, mode=mode)

    return DecomonModel(
        inputs=inputs,
        outputs=output_fusion,
        perturbation_domain=model.perturbation_domain,
        ibp=ibp,
        affine=affine,
        finetune=model.finetune,
    )
