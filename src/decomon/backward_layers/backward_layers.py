from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d_transpose
from tensorflow.keras.layers import Flatten, Layer
from tensorflow.python.ops import array_ops

from decomon.backward_layers.activations import get
from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import get_affine, get_ibp, get_identity_lirpa
from decomon.layers.convert import to_decomon
from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.decomon_layers import DecomonBatchNormalization
from decomon.layers.utils import ClipAlpha, NonNeg, NonPos
from decomon.models.utils import get_input_dim
from decomon.utils import ConvexDomainType, Option, Slope


class BackwardDense(BackwardLayer):
    """Backward  LiRPA of Dense"""

    def __init__(
        self,
        layer: Layer,
        input_dim: int = -1,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.kernel = self.layer.kernel
        self.use_bias = self.layer.use_bias
        if self.layer.use_bias:
            self.bias = self.layer.bias
        if not isinstance(self.layer, DecomonLayer):
            if input_dim < 0:
                input_dim = get_input_dim(self.layer)
            self.layer = to_decomon(
                layer=layer,
                input_dim=input_dim,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                ibp=get_ibp(self.mode),
                affine=get_affine(self.mode),
                shared=True,
                fast=False,
            )
        self.frozen_weights = False

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

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

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """
        self._trainable_weights = [self.kernel]
        if self.use_bias:
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
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
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
                convex_domain=self.convex_domain,
                finetune=False,
                ibp=get_ibp(self.mode),
                affine=get_affine(self.mode),
                shared=True,
                fast=False,
            )
        self.frozen_weights = False

    def get_bounds_linear(
        self, w_out_u: tf.Tensor, b_out_u: tf.Tensor, w_out_l: tf.Tensor, b_out_l: tf.Tensor
    ) -> List[tf.Tensor]:

        output_shape_tensor = self.layer.output_shape[-1]
        shape_ = list(output_shape_tensor)
        shape_[0] = -1
        n_out = w_out_u.shape[-1]

        # first permute dimensions
        if len(w_out_u.shape) == 2:
            w_out_u = tf.linalg.diag(w_out_u)

        if len(w_out_l.shape) == 2:
            w_out_l = tf.linalg.diag(w_out_l)

        w_out_u = array_ops.transpose(w_out_u, perm=(0, 2, 1))
        w_out_l = array_ops.transpose(w_out_l, perm=(0, 2, 1))

        if self.layer.data_format == "channels_last":
            height, width, channel = [e for e in output_shape_tensor[1:]]
            w_out_u = K.reshape(w_out_u, [-1, n_out, height, width, channel])
            w_out_l = K.reshape(w_out_l, [-1, n_out, height, width, channel])

            # start with bias
            if self.layer.use_bias:
                bias = self.layer.bias[None, None, None, None]  # (1, 1, 1, 1, channel)
                # w_out_u * bias : (None, n_out, height, width, channel)
                b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
                b_out_u_ += b_out_u
                b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
                b_out_l_ += b_out_l
            else:
                b_out_u_ = b_out_u
                b_out_l_ = b_out_l

        else:
            channel, height, width = [e for e in output_shape_tensor[1:]]
            w_out_u = K.reshape(w_out_u, [-1, n_out, channel, height, width])
            w_out_l = K.reshape(w_out_l, [-1, n_out, channel, height, width])

            # start with bias
            if self.layer.use_bias:
                bias = self.layer.bias[None, None, :, None, None]  # (1, 1, channel, 1, 1)
                # w_out_u * bias : (None, n_out, height, width, channel)
                b_out_u_ = K.sum(w_out_u * bias, (2, 3, 4))
                b_out_u_ += b_out_u
                b_out_l_ = K.sum(w_out_l * bias, (2, 3, 4))
                b_out_l_ += b_out_l
            else:
                b_out_u_ = b_out_u
                b_out_l_ = b_out_l

        def step_func(z: tf.Tensor, _: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
            return (
                conv2d_transpose(
                    z,
                    self.layer.kernel,
                    self.layer.get_input_shape_at(0)[-1],
                    strides=self.layer.strides,
                    padding=self.layer.padding,
                    data_format=self.layer.data_format,
                    dilation_rate=self.layer.dilation_rate,
                ),
                [],
            )

        step_func(w_out_u[:, 0], [])  # init
        w_out_u_ = K.rnn(step_function=step_func, inputs=w_out_u, initial_states=[], unroll=False)[1]
        w_out_l_ = K.rnn(step_function=step_func, inputs=w_out_l, initial_states=[], unroll=False)[1]

        n_in = np.prod(w_out_u_.shape[2:])
        w_out_u_ = array_ops.transpose(K.reshape(w_out_u_, [-1, n_out, n_in]), perm=(0, 2, 1))
        w_out_l_ = array_ops.transpose(K.reshape(w_out_l_, [-1, n_out, n_in]), perm=(0, 2, 1))

        return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        weight, bias = self.layer.get_backward_weights(inputs)

        z_value = K.cast(0.0, self.dtype)
        y = inputs[-1]
        shape = np.prod(y.shape[1:])
        y_flatten = K.reshape(z_value * y, (-1, np.prod(shape), 1))  # (None, n_in, 1)
        w_u_out = y_flatten + K.expand_dims(weight, 0)
        w_l_out = w_u_out
        b_u_out = K.sum(y_flatten, 1) + bias
        b_l_out = b_u_out

        return [w_u_out, b_u_out, w_l_out, b_l_out]

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
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.activation = get(layer.get_config()["activation"])
        self.activation_name = layer.get_config()["activation"]
        self.slope = Slope(slope)
        self.finetune = finetune
        self.finetune_param: List[tf.Variable] = []
        if self.finetune:
            self.frozen_alpha = False
        self.grid_finetune: List[tf.Variable] = []
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

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape: list of input shape

        Returns:

        """
        input_dim = np.prod(input_shape[-1][1:])

        if self.finetune and self.activation_name != "linear":

            if len(self.convex_domain) and ConvexDomainType(self.convex_domain["name"]) == ConvexDomainType.GRID:
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
                    K.set_value(self.alpha_b_l, alpha_b_l)
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
                K.set_value(self.alpha_b_l, alpha_b_l)

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
                len(self.convex_domain)
                and ConvexDomainType(self.convex_domain["name"]) == ConvexDomainType.GRID
                and Option(self.convex_domain["option"]) == Option.lagrangian
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
            len(self.convex_domain)
            and ConvexDomainType(self.convex_domain["name"]) == ConvexDomainType.GRID
            and Option(self.convex_domain["option"]) == Option.milp
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

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        # infer the output dimension
        if self.activation_name != "linear":
            if self.finetune:
                w_u_out, b_u_out, w_l_out, b_l_out = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune=self.finetune_param,
                    finetune_grid=self.grid_finetune,
                )
            else:
                w_u_out, b_u_out, w_l_out, b_l_out = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune_grid=self.grid_finetune,
                )
        else:
            y = inputs[-1]
            shape = np.prod(y.shape[1:])

            z_value = K.cast(0.0, self.dtype)
            o_value = K.cast(1.0, self.dtype)
            y_flat = K.reshape(y, [-1, shape])

            w_u_out, w_l_out = [o_value + z_value * y_flat] * 2
            b_u_out, b_l_out = [z_value * y_flat] * 2

            w_u_out = tf.linalg.diag(w_u_out)
            w_l_out = tf.linalg.diag(w_l_out)

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
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        y = inputs[-1]
        shape = np.prod(y.shape[1:])

        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)
        y_flat = K.reshape(y, [-1, shape])

        w_u_out, w_l_out = [o_value + z_value * y_flat] * 2
        b_u_out, b_l_out = [z_value * y_flat] * 2
        w_u_out = tf.linalg.diag(w_u_out)
        w_l_out = tf.linalg.diag(w_l_out)

        return [w_u_out, b_u_out, w_l_out, b_l_out]


class BackwardReshape(BackwardLayer):
    """Backward  LiRPA of Reshape"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        y = inputs[-1]
        shape = np.prod(y.shape[1:])

        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)
        y_flat = K.reshape(y, [-1, shape])

        w_u_out, w_l_out = [o_value + z_value * y_flat] * 2
        b_u_out, b_l_out = [z_value * y_flat] * 2
        w_u_out = tf.linalg.diag(w_u_out)
        w_l_out = tf.linalg.diag(w_l_out)

        return [w_u_out, b_u_out, w_l_out, b_l_out]


class BackwardPermute(BackwardLayer):
    """Backward LiRPA of Permute"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.dims = layer.dims
        self.op = layer.call

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        y = inputs[-1]
        shape = np.prod(y.shape[1:])
        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)
        y_flat = K.reshape(y, [-1, shape])

        w_u_out, w_l_out = [o_value + z_value * y_flat] * 2
        b_u_out, b_l_out = [z_value * y_flat] * 2
        w_u_out = tf.linalg.diag(w_u_out)
        w_l_out = tf.linalg.diag(w_l_out)

        # w_u_out (None, n_in, n_out)

        n_dim = w_u_out.shape[1]
        n_out = w_u_out.shape[-1]
        shape = list(y.shape[1:])

        w_u_out = K.reshape(w_u_out, [-1] + shape + [n_out])
        w_l_out = K.reshape(w_l_out, [-1] + shape + [n_out])

        dims = [0] + list(self.dims) + [len(y.shape)]
        dims = list(np.argsort(dims))
        w_u_out = K.reshape(K.permute_dimensions(w_u_out, dims), (-1, n_dim, n_out))
        w_l_out = K.reshape(K.permute_dimensions(w_l_out, dims), (-1, n_dim, n_out))

        return [w_u_out, b_u_out, w_l_out, b_l_out]


class BackwardDropout(BackwardLayer):
    """Backward  LiRPA of Dropout"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        y = inputs[-1]
        shape = np.prod(y.shape[1:])

        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)
        y_flat = K.reshape(y, [-1, shape])

        w_u_out, w_l_out = [o_value + z_value * y_flat] * 2
        b_u_out, b_l_out = [z_value * y_flat] * 2
        w_u_out = tf.linalg.diag(w_u_out)
        w_l_out = tf.linalg.diag(w_l_out)

        return [w_u_out, b_u_out, w_l_out, b_l_out]


class BackwardBatchNormalization(BackwardLayer):
    """Backward  LiRPA of Batch Normalization"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

        if not isinstance(layer, DecomonBatchNormalization):
            raise NotImplementedError()
        self.axis = self.layer.axis
        self.op_flat = Flatten()

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        y = inputs[0]
        w_u_out, b_u_out, w_l_out, b_l_out = inputs[-4:]

        n_dim = y.shape[1:]
        n_out = w_u_out.shape[-1]
        # reshape
        w_u_out = K.reshape(w_u_out, [-1, 1] + list(n_dim) + [n_out])
        w_l_out = K.reshape(w_l_out, [-1, 1] + list(n_dim) + [n_out])

        n_dim = len(y.shape)
        shape = [1] * n_dim
        for i, ax in enumerate(self.axis):
            shape[ax] = self.layer.moving_mean.shape[i]

        gamma = K.reshape(self.layer.gamma + 0.0, shape)
        beta = K.reshape(self.layer.beta + 0.0, shape)
        moving_mean = K.reshape(self.layer.moving_mean + 0.0, shape)
        moving_variance = K.reshape(self.layer.moving_variance + 0.0, shape)

        w = gamma / K.sqrt(moving_variance + self.layer.epsilon)
        b = beta - w * moving_mean

        # flatten w_, b_
        w = K.expand_dims(K.expand_dims(w, -1), 1)
        b = K.expand_dims(K.expand_dims(b, -1), 1)

        n_dim = np.prod(y.shape[1:])
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
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        return get_identity_lirpa(inputs)
