import logging

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Wrapper

from decomon.backward_layers.activations import get
from decomon.backward_layers.utils import get_FORWARD, get_IBP
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import to_decomon
from decomon.utils import Slope

logger = logging.getLogger(__name__)

try:
    from deel.lip.activations import GroupSort, GroupSort2
except ImportError:
    logger.warning(
        "Could not import GroupSort or GroupSort2 from deel.lip.activations. "
        "Please install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)"
    )


class BackwardDense(Wrapper):
    """Backward  LiRPA of Dense"""

    def __init__(
        self,
        layer,
        slope=Slope.V_SLOPE,
        previous=True,
        mode=ForwardMode.HYBRID,
        convex_domain=None,
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if convex_domain is None:
            convex_domain = {}
        self.layer = layer
        self.activation = get(layer.get_config()["activation"])
        self.activation_name = layer.get_config()["activation"]
        self.slope = Slope(slope)
        self.finetune = finetune
        self.previous = previous
        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = ForwardMode(mode)
            self.convex_domain = convex_domain
            self.layer = to_decomon(
                layer,
                input_dim,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                IBP=get_IBP(self.mode),
                forward=get_FORWARD(self.mode),
                shared=True,
                fast=False,
            )[0]

        self.frozen_weights = False


class BackwardGroupSort2(Wrapper):
    """Backward LiRPA of GroupSort2"""

    def __init__(
        self,
        layer,
        slope=Slope.V_SLOPE,
        previous=True,
        mode=ForwardMode.HYBRID,
        convex_domain=None,
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super().__init__(layer, kwargs)
        if convex_domain is None:
            convex_domain = {}
        self.slope = Slope(slope)
        self.finetune = finetune

        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = ForwardMode(mode)
            self.convex_domain = convex_domain
            self.layer = to_decomon(
                layer,
                input_dim,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                IBP=get_IBP(self.mode),
                forward=get_FORWARD(self.mode),
                shared=True,
                fast=False,
            )[0]

        self.frozen_weights = False

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "slope": self.slope,
                "finetune": self.finetune,
            }
        )
        return config

    def call_previous(self, inputs):

        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        if self.layer.data_format == "channels_last":
            shape = int(w_out_u.shape[1] / 2)
            n_out = w_out_u.shape[-1]
            w_out_u_ = K.reshape(w_out_u, [-1, shape, self.layer.n, n_out])
