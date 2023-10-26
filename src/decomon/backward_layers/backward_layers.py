from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Layer

from decomon.backward_layers.activations import get
from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import get_identity_lirpa
from decomon.backward_layers.utils_conv import get_toeplitz
from decomon.core import (
    ForwardMode,
    GridDomain,
    Option,
    PerturbationDomain,
    Slope,
    get_affine,
    get_ibp,
)
from decomon.layers.convert import to_decomon
from decomon.layers.core import DecomonLayer
from decomon.layers.decomon_layers import DecomonBatchNormalization
from decomon.layers.utils import ClipAlpha, NonNeg, NonPos
from decomon.models.utils import get_input_dim


class BackwardDense(BackwardLayer):
    """Backward  LiRPA of Dense"""

    def __init__(
        self,
        layer: Layer,
        input_dim: int = -1,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.use_bias = self.layer.use_bias
        if not isinstance(self.layer, DecomonLayer):
            if input_dim < 0:
                input_dim = get_input_dim(self.layer)
            self.layer = to_decomon(
                layer=layer,
                input_dim=input_dim,
                dc_decomp=False,
                perturbation_domain=self.perturbation_domain,
                finetune=False,
                ibp=get_ibp(self.mode),
                affine=get_affine(self.mode),
                shared=True,
                fast=False,
            )
        self.frozen_weights = False

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        if len(inputs) == 0:
            inputs = self.layer.input

        weights = self.kernel
        y = inputs[-1]
        z_value = K.cast(0.0, self.dtype)
        w_u_out, w_l_out = [weights[None] + z_value * K.expand_dims(y, -1)] * 2
        if self.use_bias:
            bias = self.bias
            b_u_out, b_l_out = [bias[None] + z_value * w_u_out[:, 0]] * 2
        else:
            b_u_out, b_l_out = [z_value * w_u_out[:, 0]] * 2
        return [w_u_out, b_u_out, w_l_out, b_l_out]

    def build(self, input_shape: List[Tuple[Optional[int]]]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """
        if self.layer.kernel is None:
            raise RuntimeError("self.layer.kernel cannot be None when calling self.build()")
        self.kernel = self.layer.kernel
        self._trainable_weights = [self.kernel]
        if self.use_bias:
            if self.layer.bias is None:
                raise RuntimeError("self.layer.bias cannot be None when calling self.build()")
            self.bias = self.layer.bias
            self._trainable_weights.append(self.bias)
        self.built = True

    def freeze_weights(self) -> None:
        if not self.frozen_weights:
            self._trainable_weights = []
            if getattr(self.layer, "freeze_weights"):
                self.layer.freeze_weights()
            self.frozen_weights = True

    def unfreeze_weights(self) -> None:
        if self.frozen_weights:
            if getattr(self.layer, "unfreeze_weights"):
                self.layer.unfreeze_weights()
            self.frozen_weights = False


class BackwardConv2D(BackwardLayer):
    """Backward  LiRPA of Conv2D"""

    def __init__(
        self,
        layer: Layer,
        input_dim: int = -1,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(self.layer, DecomonLayer):
            if input_dim < 0:
                input_dim = get_input_dim(self.layer)
            self.layer = to_decomon(
                layer=layer,
                input_dim=input_dim,
                dc_decomp=False,
                perturbation_domain=self.perturbation_domain,
                finetune=False,
                ibp=get_ibp(self.mode),
                affine=get_affine(self.mode),
                shared=True,
                fast=False,
            )
        self.frozen_weights = False

    def get_affine_components(self, inputs: List[keras.KerasTensor]) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Express the implicit affine matrix of the convolution layer.

        Conv is a linear operator but its affine component is implicit
        we use im2col and extract_patches to express the affine matrix
        Note that this matrix is Toeplitz

        Args:
            inputs: list of input tensors
        Returns:
            the affine operators W, b : conv(inputs)= W.inputs + b
        """

        w_out_u_ = get_toeplitz(self.layer, True)
        output_shape = self.layer.get_output_shape_at(0)
        if isinstance(output_shape, list):
            output_shape = output_shape[-1]
        output_shape = output_shape[1:]
        if self.layer.data_format == "channels_last":
            b_out_u_ = K.reshape(K.zeros(output_shape, dtype=self.layer.dtype), (-1, output_shape[-1]))
        else:
            b_out_u_ = K.transpose(
                K.reshape(K.zeros(output_shape, dtype=self.layer.dtype), (-1, output_shape[0])), (1, 0)
            )

        if self.layer.use_bias:
            bias_ = K.cast(self.layer.bias, self.layer.dtype)
            b_out_u_ = b_out_u_ + bias_[None]
        b_out_u_ = K.ravel(b_out_u_)

        z_value = K.cast(0.0, self.dtype)
        y_ = inputs[-1]
        shape = int(np.prod(y_.shape[1:]))
        y_flatten = K.reshape(z_value * y_, (-1, int(np.prod(shape))))  # (None, n_in)
        w_out_ = K.sum(y_flatten, -1)[:, None, None] + w_out_u_
        b_out_ = K.sum(y_flatten, -1)[:, None] + b_out_u_

        return w_out_, b_out_

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        weight_, bias_ = self.get_affine_components(inputs)
        return [weight_, bias_] * 2

    def freeze_weights(self) -> None:
        if not self.frozen_weights:
            self._trainable_weights = []
            if getattr(self.layer, "freeze_weights"):
                self.layer.freeze_weights()
            self.frozen_weights = True

    def unfreeze_weights(self) -> None:
        if self.frozen_weights:
            if getattr(self.layer, "unfreeze_weights"):
                self.layer.unfreeze_weights()
            self.frozen_weights = False


class BackwardActivation(BackwardLayer):
    def __init__(
        self,
        layer: Layer,
        slope: Union[str, Slope] = Slope.V_SLOPE,
        finetune: bool = False,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.activation = get(layer.get_config()["activation"])
        self.activation_name = layer.get_config()["activation"]
        self.slope = Slope(slope)
        self.finetune = finetune
        self.finetune_param: List[keras.Variable] = []
        if self.finetune:
            self.frozen_alpha = False
        self.grid_finetune: List[keras.Variable] = []
        self.frozen_grid = False

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "slope": self.slope,
                "finetune": self.finetune,
            }
        )
        return config

    def build(self, input_shape: List[Tuple[Optional[int]]]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """
        input_dim = int(np.prod(input_shape[-1][1:]))

        if self.finetune and self.activation_name != "linear":
            if isinstance(self.perturbation_domain, GridDomain):
                if self.activation_name[:4] == "relu":
                    self.alpha_b_l = self.add_weight(
                        shape=(
                            3,
                            input_dim,
                        ),
                        initializer="ones",
                        name="alpha_l_b_0",
                        regularizer=None,
                        constraint=ClipAlpha(),
                    )
                    alpha_b_l = np.zeros((3, input_dim))
                    alpha_b_l[0] = 1
                    self.alpha_b_l.assign(alpha_b_l)
                    self.finetune_param.append(self.alpha_b_l)

            else:
                self.alpha_b_l = self.add_weight(
                    shape=(input_dim,),
                    initializer="ones",
                    name="alpha_l_b",
                    regularizer=None,
                    constraint=ClipAlpha(),
                )
                alpha_b_l = np.ones((input_dim,))
                # alpha_b_l[0] = 1
                self.alpha_b_l.assign(alpha_b_l)

                if self.activation_name[:4] != "relu":
                    self.alpha_b_u = self.add_weight(
                        shape=(input_dim,),
                        initializer="ones",
                        name="alpha_u_b",
                        regularizer=None,
                        constraint=ClipAlpha(),
                    )
                    self.finetune_param.append(self.alpha_b_u)

                self.finetune_param.append(self.alpha_b_l)
            if len(self.finetune_param) == 1:
                self.finetune_param = self.finetune_param[0]

        # grid domain
        if self.activation_name[:4] == "relu":
            if (
                isinstance(self.perturbation_domain, GridDomain)
                and self.perturbation_domain.opt_option == Option.lagrangian
                and self.mode != ForwardMode.IBP
            ):
                finetune_grid_pos = self.add_weight(
                    shape=(input_dim,),
                    initializer="zeros",
                    name="lambda_grid_neg",
                    regularizer=None,
                    constraint=NonNeg(),
                )

                finetune_grid_neg = self.add_weight(
                    shape=(input_dim,),
                    initializer="zeros",
                    name="lambda_grid_pos",
                    regularizer=None,
                    constraint=NonPos(),
                )

                self.grid_finetune = [finetune_grid_neg, finetune_grid_pos]

        if (
            isinstance(self.perturbation_domain, GridDomain)
            and self.perturbation_domain.opt_option == Option.milp
            and self.mode != ForwardMode.IBP
        ):
            finetune_grid_A = self.add_weight(
                shape=(input_dim,),
                initializer="zeros",
                name=f"A_{self.layer.name}_{self.rec}",
                regularizer=None,
                trainable=False,
            )  # constraint=NonPos()
            finetune_grid_B = self.add_weight(
                shape=(input_dim,),
                initializer="zeros",
                name=f"B_{self.layer.name}_{self.rec}",
                regularizer=None,
                trainable=False,
            )  # constraint=NonNeg()

            self.grid_finetune = [finetune_grid_A, finetune_grid_B]

        self.built = True

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        # infer the output dimension
        if self.activation_name != "linear":
            if self.finetune:
                w_u_out, b_u_out, w_l_out, b_l_out = self.activation(
                    inputs,
                    perturbation_domain=self.perturbation_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune=self.finetune_param,
                    finetune_grid=self.grid_finetune,
                )
            else:
                w_u_out, b_u_out, w_l_out, b_l_out = self.activation(
                    inputs,
                    perturbation_domain=self.perturbation_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune_grid=self.grid_finetune,
                )
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        if len(w_u_out.shape) == 2:
            w_u_out = tf.linalg.diag(w_u_out)
        if len(w_l_out.shape) == 2:
            w_l_out = tf.linalg.diag(w_l_out)

        return [w_u_out, b_u_out, w_l_out, b_l_out]

    def freeze_alpha(self) -> None:
        if not self.frozen_alpha:
            if self.finetune and self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                if len(self.grid_finetune):
                    self._trainable_weights = self._trainable_weights[:2]
                else:
                    self._trainable_weights = []
                self.frozen_alpha = True

    def unfreeze_alpha(self) -> None:
        if self.frozen_alpha:
            if self.finetune and self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
                if self.activation_name != "linear":
                    if self.activation_name[:4] != "relu":
                        self._trainable_weights += [self.alpha_b_u, self.alpha_b_l]
                    else:
                        self._trainable_weights += [self.alpha_b_l]
            self.frozen_alpha = False

    def freeze_grid(self) -> None:
        if len(self.grid_finetune) and not self.frozen_grid:
            self._trainable_weights = self._trainable_weights[2:]
            self.frozen_grid = True

    def unfreeze_grid(self) -> None:
        if len(self.grid_finetune) and self.frozen_grid:
            self._trainable_weights = self.grid_finetune + self._trainable_weights
            self.frozen_grid = False


class BackwardFlatten(BackwardLayer):
    """Backward  LiRPA of Flatten"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        return get_identity_lirpa(inputs)


class BackwardReshape(BackwardLayer):
    """Backward  LiRPA of Reshape"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        return get_identity_lirpa(inputs)


class BackwardPermute(BackwardLayer):
    """Backward LiRPA of Permute"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.dims = layer.dims
        self.op = layer.call

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        # w_u_out (None, n_in, n_out)
        y = inputs[-1]
        n_dim = w_u_out.shape[1]
        n_out = w_u_out.shape[-1]
        shape = list(y.shape[1:])

        w_u_out = K.reshape(w_u_out, [-1] + shape + [n_out])
        w_l_out = K.reshape(w_l_out, [-1] + shape + [n_out])

        dims = [0] + list(self.dims) + [len(y.shape)]
        dims = list(np.argsort(dims))
        w_u_out = K.reshape(K.transpose(w_u_out, dims), (-1, n_dim, n_out))
        w_l_out = K.reshape(K.transpose(w_l_out, dims), (-1, n_dim, n_out))

        return [w_u_out, b_u_out, w_l_out, b_l_out]


class BackwardDropout(BackwardLayer):
    """Backward  LiRPA of Dropout"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        return get_identity_lirpa(inputs)


class BackwardBatchNormalization(BackwardLayer):
    """Backward  LiRPA of Batch Normalization"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

        if not isinstance(layer, DecomonBatchNormalization):
            raise NotImplementedError()
        self.axis = self.layer.axis
        self.op_flat = Flatten()

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        y = inputs[-1]
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        n_dim = y.shape[1:]
        n_out = w_u_out.shape[-1]
        # reshape
        w_u_out = K.reshape(w_u_out, [-1, 1] + list(n_dim) + [n_out])
        w_l_out = K.reshape(w_l_out, [-1, 1] + list(n_dim) + [n_out])

        n_dim = len(y.shape)
        shape = [1] * n_dim
        shape[self.axis] = self.layer.moving_mean.shape[0]

        if not hasattr(self.layer, "gamma") or self.layer.gamma is None:  # scale = False
            gamma = K.ones(shape)
        else:  # scale = True
            gamma = K.reshape(self.layer.gamma + 0.0, shape)
        if not hasattr(self.layer, "beta") or self.layer.beta is None:  # center = False
            beta = K.zeros(shape)
        else:  # center = True
            beta = K.reshape(self.layer.beta + 0.0, shape)
        moving_mean = K.reshape(self.layer.moving_mean + 0.0, shape)
        moving_variance = K.reshape(self.layer.moving_variance + 0.0, shape)

        w = gamma / K.sqrt(moving_variance + self.layer.epsilon)
        b = beta - w * moving_mean

        # flatten w_, b_
        w = K.expand_dims(K.expand_dims(w, -1), 1)
        b = K.expand_dims(K.expand_dims(b, -1), 1)

        n_dim = int(np.prod(y.shape[1:]))
        w_u_b = K.reshape(w_u_out * w, (-1, n_dim, n_out))
        w_l_b = K.reshape(w_l_out * w, (-1, n_dim, n_out))
        axis = [i for i in range(2, len(b.shape) - 1)]
        b_u_b = K.sum(w_u_out * b, axis) + b_u_out
        b_l_b = K.sum(w_l_out * b, axis) + b_l_out

        return [w_u_b, b_u_b, w_l_b, b_l_b]


class BackwardInputLayer(BackwardLayer):
    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        return get_identity_lirpa(inputs)
