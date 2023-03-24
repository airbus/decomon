import logging
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer

from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.decomon_merge_layers import DecomonConcatenate
from decomon.layers.decomon_reshape import DecomonReshape
from decomon.layers.utils import ClipAlpha, expand_dims, max_, min_, sort

logger = logging.getLogger(__name__)

try:
    from deel.lip.activations import GroupSort, GroupSort2
except ImportError:
    logger.warning(
        "Could not import GroupSort or GroupSort2 from deel.lip.activations. "
        "Please install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)"
    )


class DecomonGroupSort(DecomonLayer):
    def __init__(
        self,
        n: Optional[int] = None,
        data_format: str = "channels_last",
        k_coef_lip: float = 1.0,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        **kwargs: Any,
    ):
        super().__init__(mode=mode, **kwargs)
        self.data_format = data_format
        if data_format == "channels_last":
            self.channel_axis = -1
        elif data_format == "channels_first":
            raise RuntimeError("channels_first not implemented for GroupSort activation")
        else:
            raise RuntimeError("data format not understood")
        self.n = n
        self.concat = DecomonConcatenate(
            mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "mode": self.mode,
                "n": self.n,
            }
        )
        return config

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        if (self.n is None) or (self.n > input_shape[-1][self.channel_axis]):
            self.n = input_shape[-1][self.channel_axis]
            if self.n is None:  # for mypy
                raise RuntimeError("self.n cannot be None at this point.")
        self.reshape = DecomonReshape(
            (-1, self.n), mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        shape_in = tuple(inputs[-1].shape[1:])
        input_ = self.reshape(inputs)
        if self.n == 2:

            output_max = expand_dims(
                max_(input_, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode, axis=-1),
                dc_decomp=self.dc_decomp,
                mode=self.mode,
                axis=-1,
            )
            output_min = expand_dims(
                min_(input_, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode, axis=-1),
                dc_decomp=self.dc_decomp,
                mode=self.mode,
                axis=-1,
            )
            output_ = self.concat(output_min + output_max)

        else:

            output_ = sort(input_, axis=-1, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode)

        return DecomonReshape(
            shape_in, mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call(output_)

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        return input_shape


class DecomonGroupSort2(DecomonLayer):
    def __init__(
        self,
        n: int = 2,
        data_format: str = "channels_last",
        k_coef_lip: float = 1.0,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        **kwargs: Any,
    ):
        super().__init__(mode=mode, **kwargs)
        self.data_format = data_format

        if self.data_format == "channels_last":
            self.axis = -1
        else:
            self.axis = 1

        if self.dc_decomp:
            raise NotImplementedError()

        self.op_concat = DecomonConcatenate(self.axis, mode=self.mode, convex_domain=self.convex_domain)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "mode": self.mode,
            }
        )
        return config

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        return input_shape

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        inputs_ = self.op_reshape_in(inputs)
        inputs_max = expand_dims(
            max_(
                inputs_,
                mode=self.mode,
                convex_domain=self.convex_domain,
                axis=self.axis,
                finetune=self.finetune,
                finetune_params=self.params_max,
            ),
            mode=self.mode,
            axis=self.axis,
        )
        inputs_min = expand_dims(
            min_(
                inputs_,
                mode=self.mode,
                convex_domain=self.convex_domain,
                axis=self.axis,
                finetune=self.finetune,
                finetune_params=self.params_min,
            ),
            mode=self.mode,
            axis=self.axis,
        )
        output = self.op_concat(inputs_min + inputs_max)
        output_ = self.op_reshape_out(output)
        return output_

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        input_shape = input_shape[-1]

        if self.data_format == "channels_last":
            if input_shape[-1] % 2 != 0:
                raise ValueError()
            target_shape = list(input_shape[1:-2]) + [int(input_shape[-1] / 2), 2]
        else:
            if input_shape[1] % 2 != 0:
                raise ValueError()
            target_shape = [2, int(input_shape[1] / 2)] + list(input_shape[2:])

        self.params_max = []
        self.params_min = []

        if self.finetune and self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            self.beta_max_ = self.add_weight(
                shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
            )
            self.beta_min_ = self.add_weight(
                shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
            )
            self.params_max = [self.beta_max_]
            self.params_min = [self.beta_min_]

        self.op_reshape_in = DecomonReshape(tuple(target_shape), mode=self.mode)
        self.op_reshape_out = DecomonReshape(tuple(input_shape[1:]), mode=self.mode)

    def reset_layer(self, layer: Layer) -> None:
        pass
