import logging

from decomon.layers.core import F_FORWARD, F_HYBRID, DecomonLayer
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
    def __init__(self, n=None, data_format="channels_last", k_coef_lip=1.0, mode=F_HYBRID.name, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.data_format = data_format
        if data_format == "channels_last":
            self.channel_axis = -1
        elif data_format == "channels_first":
            raise RuntimeError("channels_first not implemented for GroupSort activation")
        else:
            raise RuntimeError("data format not understood")
        self.n = n
        self.reshape = DecomonReshape(
            (-1, self.n), mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call
        self.concat = DecomonConcatenate(
            mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "mode": self.mode,
                "n": self.n,
            }
        )
        return config

    def call(self, input, **kwargs):

        shape_in = list(input[0].shape[1:])
        input_ = self.reshape(input)
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
            output_ = self.concat([output_min, output_max])

        else:

            output_ = sort(input_, axis=-1, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode)

        return DecomonReshape(
            shape_in, mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call(output_)

    def compute_output_shape(self, input_shape):
        return input_shape


class DecomonGroupSort2(DecomonLayer):
    def __init__(self, n=2, data_format="channels_last", k_coef_lip=1.0, mode=F_HYBRID.name, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.data_format = data_format

        if self.data_format == "channels_last":
            self.axis = -1
        else:
            self.axis = 1

        if self.dc_decomp:
            raise NotImplementedError()

        self.op_concat = DecomonConcatenate(self.axis, mode=self.mode, convex_domain=self.convex_domain)
        self.op_reshape_in = None
        self.op_reshape_out = None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "mode": self.mode,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):

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

    def build(self, input_shape):
        input_shape = input_shape[-1]

        if self.data_format == "channels_last":
            if input_shape[-1] % 2 != 0:
                raise ValueError()
            target_shape = input_shape[1:-2] + [int(input_shape[-1] / 2), 2]
        else:
            if input_shape[1] % 2 != 0:
                raise ValueError()
            target_shape = [2, int(input_shape[1] / 2)] + input_shape[2:]

        self.params_max = []
        self.params_min = []

        if self.finetune and self.mode in [F_FORWARD.name, F_HYBRID.name]:
            self.beta_max_ = self.add_weight(
                shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
            )
            self.beta_min_ = self.add_weight(
                shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
            )
            self.params_max = [self.beta_max_]
            self.params_min = [self.beta_min_]

        self.op_reshape_in = DecomonReshape(target_shape, mode=self.mode)
        self.op_reshape_out = DecomonReshape(input_shape[1:], mode=self.mode)

    def reset_layer(self, layer):
        pass
