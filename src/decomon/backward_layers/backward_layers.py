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
from decomon.layers.core import DecomonLayer, ForwardMode, Option
from decomon.layers.decomon_layers import DecomonBatchNormalization
from decomon.layers.utils import ClipAlpha, NonNeg, NonPos
from decomon.models.utils import get_input_dim
from decomon.utils import ConvexDomainType, Slope


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

        if len(inputs):
            x_ = inputs
        else:
            x_ = self.layer.input

        # start with the activation: determine the upper and lower bounds before the weights
        weights = self.kernel
        y_ = x_[-1]
        z_value = K.cast(0.0, self.dtype)
        w_out_u_, w_out_l_ = [weights[None] + z_value * K.expand_dims(y_, -1)] * 2
        if self.use_bias:
            bias = self.bias
            b_out_u_, b_out_l_ = [bias[None] + z_value * w_out_u_[:, 0]] * 2
        else:
            b_out_u_, b_out_l_ = [z_value * w_out_u_[:, 0]] * 2
        return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]

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
        x = inputs

        weight, bias = self.layer.get_backward_weights(x)

        z_value = K.cast(0.0, self.dtype)
        y_ = x[-1]
        shape = np.prod(y_.shape[1:])
        y_flatten = K.reshape(z_value * y_, (-1, np.prod(shape), 1))  # (None, n_in, 1)
        w_out_u_ = y_flatten + K.expand_dims(weight, 0)
        w_out_l_ = w_out_u_
        b_out_u_ = K.sum(y_flatten, 1) + bias
        b_out_l_ = b_out_u_

        return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]

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

            if len(self.convex_domain) and self.convex_domain["name"] == ConvexDomainType.GRID:
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
                and self.convex_domain["name"] == ConvexDomainType.GRID
                and self.convex_domain["option"] == Option.lagrangian
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
            and self.convex_domain["name"] == ConvexDomainType.GRID
            and self.convex_domain["option"] == Option.milp
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
        y_ = inputs[-1]
        if self.activation_name != "linear":
            if self.finetune:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune=self.finetune_param,
                    finetune_grid=self.grid_finetune,
                )
            else:
                w_out_u, b_out_u, w_out_l, b_out_l = self.activation(
                    inputs,
                    convex_domain=self.convex_domain,
                    slope=self.slope,
                    mode=self.mode,
                    finetune_grid=self.grid_finetune,
                )
        else:
            y_ = inputs[-1]
            shape = np.prod(y_.shape[1:])

            z_value = K.cast(0.0, self.dtype)
            o_value = K.cast(1.0, self.dtype)
            y_flat = K.reshape(y_, [-1, shape])

            w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
            b_out_u, b_out_l = [z_value * y_flat] * 2

            w_out_u = tf.linalg.diag(w_out_u)
            w_out_l = tf.linalg.diag(w_out_l)

        if len(w_out_u.shape) == 2:
            w_out_u = tf.linalg.diag(w_out_u)
        if len(w_out_l.shape) == 2:
            w_out_l = tf.linalg.diag(w_out_l)

        return [w_out_u, b_out_u, w_out_l, b_out_l]

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
        y_ = inputs[-1]
        shape = np.prod(y_.shape[1:])

        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)
        y_flat = K.reshape(y_, [-1, shape])

        w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
        b_out_u, b_out_l = [z_value * y_flat] * 2
        w_out_u = tf.linalg.diag(w_out_u)
        w_out_l = tf.linalg.diag(w_out_l)

        return [w_out_u, b_out_u, w_out_l, b_out_l]


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

        y_ = inputs[-1]
        shape = np.prod(y_.shape[1:])

        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)
        y_flat = K.reshape(y_, [-1, shape])

        w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
        b_out_u, b_out_l = [z_value * y_flat] * 2
        w_out_u = tf.linalg.diag(w_out_u)
        w_out_l = tf.linalg.diag(w_out_l)

        return [w_out_u, b_out_u, w_out_l, b_out_l]


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

        w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
        b_out_u, b_out_l = [z_value * y_flat] * 2
        w_out_u = tf.linalg.diag(w_out_u)
        w_out_l = tf.linalg.diag(w_out_l)

        # w_out_u (None, n_in, n_out)

        n_dim = w_out_u.shape[1]
        n_out = w_out_u.shape[-1]
        shape = list(y.shape[1:])

        w_out_u_ = K.reshape(w_out_u, [-1] + shape + [n_out])
        w_out_l_ = K.reshape(w_out_l, [-1] + shape + [n_out])

        dims = [0] + list(self.dims) + [len(y.shape)]
        dims = list(np.argsort(dims))
        w_out_u_0 = K.reshape(K.permute_dimensions(w_out_u_, dims), (-1, n_dim, n_out))
        w_out_l_0 = K.reshape(K.permute_dimensions(w_out_l_, dims), (-1, n_dim, n_out))

        return [w_out_u_0, b_out_u, w_out_l_0, b_out_l]


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

        y_ = inputs[-1]
        shape = np.prod(y_.shape[1:])

        z_value = K.cast(0.0, self.dtype)
        o_value = K.cast(1.0, self.dtype)
        y_flat = K.reshape(y_, [-1, shape])

        w_out_u, w_out_l = [o_value + z_value * y_flat] * 2
        b_out_u, b_out_l = [z_value * y_flat] * 2
        w_out_u = tf.linalg.diag(w_out_u)
        w_out_l = tf.linalg.diag(w_out_l)

        return [w_out_u, b_out_u, w_out_l, b_out_l]


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
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_dim = y.shape[1:]
        n_out = w_out_u.shape[-1]
        # reshape
        w_out_u = K.reshape(w_out_u, [-1, 1] + list(n_dim) + [n_out])
        w_out_l = K.reshape(w_out_l, [-1, 1] + list(n_dim) + [n_out])

        n_dim = len(y.shape)
        tuple_ = [1] * n_dim
        for i, ax in enumerate(self.axis):
            tuple_[ax] = self.layer.moving_mean.shape[i]

        gamma_ = K.reshape(self.layer.gamma + 0.0, tuple_)
        beta_ = K.reshape(self.layer.beta + 0.0, tuple_)
        moving_mean_ = K.reshape(self.layer.moving_mean + 0.0, tuple_)
        moving_variance_ = K.reshape(self.layer.moving_variance + 0.0, tuple_)

        w_ = gamma_ / K.sqrt(moving_variance_ + self.layer.epsilon)
        b_ = beta_ - w_ * moving_mean_

        # flatten w_, b_
        w_ = K.expand_dims(K.expand_dims(w_, -1), 1)
        b_ = K.expand_dims(K.expand_dims(b_, -1), 1)

        n_dim = np.prod(y.shape[1:])
        w_u_b_ = K.reshape(w_out_u * w_, (-1, n_dim, n_out))
        w_l_b_ = K.reshape(w_out_l * w_, (-1, n_dim, n_out))
        axis = [i for i in range(2, len(b_.shape) - 1)]
        b_u_b_ = K.sum(w_out_u * b_, axis) + b_out_u
        b_l_b_ = K.sum(w_out_l * b_, axis) + b_out_l

        return [w_u_b_, b_u_b_, w_l_b_, b_l_b_]


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
