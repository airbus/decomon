from __future__ import absolute_import

try:
    from deel.lip.activations import GroupSort, GroupSort2
except:
    raise Warning('Could not import GroupSort from deel.lip.activations. Install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)')


import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten, Dot, Permute, Activation
from ..backward_layers.activations import get
from tensorflow.keras.backend import conv2d, conv2d_transpose
from .utils import V_slope, backward_sort, get_identity_lirpa, get_IBP, get_FORWARD, get_input_dim
from ..layers.utils import ClipAlpha, F_HYBRID, F_FORWARD
from ..layers.decomon_layers import to_monotonic


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
        convex_domain={},
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super(BackwardDense, self).__init__(**kwargs)

        self.layer = layer
        self.activation = get(layer.get_config()["activation"])  # ??? not sur
        self.activation_name = layer.get_config()["activation"]
        self.slope = slope
        # if hasattr(self.layer, 'finetune'):
        #    self.finetune=self.layer.finetune
        # else:
        #    self.finetune = False
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

    def __init__(self,
                 layer,
                 slope=V_slope.name,
                 previous=True,
                 mode=F_HYBRID.name,
                 convex_domain={},
                 finetune=False,
                 input_dim=-1,
                 **kwargs,
                 ):
        super(BackwardGroupSort2, self).__init__(kwargs)
        self.layer = layer
        self.slope=slope
        self.finetune=finetune

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

        x= inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        if self.layer.data_format=="channels_last":
            shape = int(w_out_u.shape[1]/2)
            n_out = w_out_u.shape[-1]
            w_out_u_ = K.reshape(w_out_u, [-1, shape, self.layer.n, n_out])
            w




