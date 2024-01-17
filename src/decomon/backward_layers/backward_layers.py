from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
import numpy as np
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
from decomon.keras_utils import BatchedDiagLike, BatchedIdentityLike
from decomon.layers.convert import to_decomon
from decomon.layers.core import DecomonLayer
from decomon.layers.decomon_layers import DecomonBatchNormalization
from decomon.layers.utils import ClipAlpha, NonNeg, NonPos
from decomon.models.utils import get_input_dim
from decomon.types import BackendTensor


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

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
        if len(inputs) == 0:
            inputs = self.layer.input

        y = inputs[-1]
        flatten_inputdim_wo_last_dim = int(np.prod(y.shape[1:-1]))
        flatten_outputdim = flatten_inputdim_wo_last_dim * self.layer.units
        batchsize = y.shape[0]

        # kernel reshaped: diagonal by blocks, with original kernel on the diagonal, `flatten_inputdim_wo_last_dim` times
        #                  repeated batchsize times
        zero_block = K.zeros_like(self.kernel)
        kernel_diag_by_block = K.concatenate(
            [
                K.concatenate(
                    [self.kernel if i == j else zero_block for i in range(flatten_inputdim_wo_last_dim)], axis=-1
                )
                for j in range(flatten_inputdim_wo_last_dim)
            ],
            axis=-2,
        )
        w = K.repeat(kernel_diag_by_block[None], batchsize, axis=0)
        if self.use_bias:
            b = K.repeat(
                K.reshape(K.repeat(self.bias[None], flatten_inputdim_wo_last_dim, axis=0), (1, -1)), batchsize, axis=0
            )
        else:
            b = K.zeros((batchsize, flatten_outputdim), dtype=self.dtype)

        return [w, b, w, b]

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
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

    def get_affine_components(self, inputs: List[BackendTensor]) -> Tuple[BackendTensor, BackendTensor]:
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
        output = self.layer.output
        if isinstance(output, keras.KerasTensor):
            output_shape = output.shape
        else:  # list of outputs
            output_shape = output[-1].shape
        output_shape = output_shape[1:]
        if self.layer.data_format == "channels_last":
            b_out_u_ = K.reshape(K.zeros(output_shape, dtype=self.layer.dtype), (-1, output_shape[-1]))
        else:
            b_out_u_ = K.transpose(
                K.reshape(K.zeros(output_shape, dtype=self.layer.dtype), (-1, output_shape[0])), (1, 0)
            )

        if self.layer.use_bias:
            bias_ = K.cast(self.layer.bias, self.layer.dtype)
            if self.layer.data_format == "channels_last":
                b_out_u_ = b_out_u_ + bias_[None]
            else:
                b_out_u_ = b_out_u_ + bias_[:, None]
        b_out_u_ = K.ravel(b_out_u_)

        z_value = K.cast(0.0, self.dtype)
        y_ = inputs[-1]
        shape = int(np.prod(y_.shape[1:]))
        y_flatten = K.reshape(z_value * y_, (-1, int(np.prod(shape))))  # (None, n_in)
        w_out_ = K.sum(y_flatten, -1)[:, None, None] + w_out_u_
        b_out_ = K.sum(y_flatten, -1)[:, None] + b_out_u_

        return w_out_, b_out_

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
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

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """
        input_dim = int(np.prod(input_shape[-1][1:]))  # type: ignore

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

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
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
            w_u_out = BatchedDiagLike()(w_u_out)
        if len(w_l_out.shape) == 2:
            w_l_out = BatchedDiagLike()(w_l_out)

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

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
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

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
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

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
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

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
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

        self.axis = self.layer.axis
        self.op_flat = Flatten()

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
        y = inputs[-1]
        n_out = int(np.prod(y.shape[1:]))

        n_dim = len(y.shape)
        shape = [1] * n_dim
        shape[self.axis] = self.layer.moving_mean.shape[0]

        if not hasattr(self.layer, "gamma") or self.layer.gamma is None:  # scale = False
            gamma = K.ones_like(self.layer.moving_variance)
        else:  # scale = True
            gamma = self.layer.gamma
        if not hasattr(self.layer, "beta") or self.layer.beta is None:  # center = False
            beta = K.zeros_like(self.layer.moving_mean)
        else:  # center = True
            beta = self.layer.beta

        w = gamma / K.sqrt(self.layer.moving_variance + self.layer.epsilon)
        b = beta - w * self.layer.moving_mean

        # reshape w
        w_b = K.reshape(
            K.reshape(BatchedIdentityLike()(K.reshape(y, (-1, n_out))), tuple(y.shape) + (-1,))
            * K.reshape(w, shape + [1]),
            (-1, n_out, n_out),
        )

        # reshape b
        b_b = K.reshape(K.ones_like(y) * K.reshape(b, shape), (-1, n_out))

        return [w_b, b_b, w_b, b_b]


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

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
        return get_identity_lirpa(inputs)
