try:
    from deel.lip.activations import GroupSort, GroupSort2
except:
    raise Warning(
        "Could not import GroupSort from deel.lip.activations. Install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)"
    )


import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d, conv2d_transpose
from tensorflow.keras.layers import Activation, Dot, Flatten, Layer, Permute

from decomon.backward_layers.activations import get
from decomon.backward_layers.utils import (
    V_slope,
    backward_sort,
    get_FORWARD,
    get_IBP,
    get_identity_lirpa,
    get_input_dim,
)
from decomon.layers.decomon_layers import to_monotonic
from decomon.layers.utils import F_FORWARD, F_HYBRID, ClipAlpha


class BackwardDense(Layer):
    """
    Backward  LiRPA of Dense
    """

    def __init__(
        self,
        layer,
        slope=V_slope.name,
        previous=True,
        mode=F_HYBRID.name,
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
        self.slope = slope
        self.finetune = finetune
        self.previous = previous
        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = mode
            self.convex_domain = convex_domain
            input_dim_ = get_input_dim(input_dim, self.convex_domain)
            self.layer = to_monotonic(
                layer,
                input_dim_,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                IBP=get_IBP(self.mode),
                forward=get_FORWARD(self.mode),
                shared=True,
                fast=False,
            )[0]

        self.frozen_weights = False


class BackwardGroupSort2(Layer):
    """
    Backward LiRPA of GroupSort2
    """

    def __init__(
        self,
        layer,
        slope=V_slope.name,
        previous=True,
        mode=F_HYBRID.name,
        convex_domain=None,
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super().__init__(kwargs)
        if convex_domain is None:
            convex_domain = {}
        self.layer = layer
        self.slope = slope
        self.finetune = finetune

        if hasattr(self.layer, "mode"):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
        else:
            self.mode = mode
            self.convex_domain = convex_domain
            input_dim_ = get_input_dim(input_dim, self.convex_domain)
            self.layer = to_monotonic(
                layer,
                input_dim_,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                IBP=get_IBP(self.mode),
                forward=get_FORWARD(self.mode),
                shared=True,
                fast=False,
            )[0]

        self.frozen_weights = False

    def call_previous(self, inputs):

        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        if self.layer.data_format == "channels_last":
            shape = int(w_out_u.shape[1] / 2)
            n_out = w_out_u.shape[-1]
            w_out_u_ = K.reshape(w_out_u, [-1, shape, self.layer.n, n_out])
