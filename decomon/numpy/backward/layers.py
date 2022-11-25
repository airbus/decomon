"""
CROWN implementation in numpy (useful for MILP compatibility)
"""
from __future__ import absolute_import

import numpy as np
from tensorflow.keras.layers import Activation, Dense, InputLayer

from decomon.utils import F_FORWARD

from .activation import get
from .core import BackwardNumpyLayer
from .utils import merge_with_previous


class BackwardNumpyActivation(BackwardNumpyLayer):
    def __init__(
        self, keras_layer, previous=False, convex_domain=None, mode=F_FORWARD.name, rec=1, params=None, **kwargs
    ):
        """

        :param convex_domain: type of convex input domain (None or dict)
        :param dc_decomp: boolean that indicates whether we return a
        difference of convex decomposition of our layer
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        :param kwargs: extra parameters
        """
        super().__init__(
            keras_layer=keras_layer, convex_domain=convex_domain, mode=mode, rec=rec, params=params, **kwargs
        )
        if convex_domain is None:
            convex_domain = {}
        if params is None:
            params = []
        self.activation_name = keras_layer.get_config()["activation"]
        self.activation = get(keras_layer.get_config()["activation"])

        self.history_x = None
        self.history_A = None
        self.history_B = None

        if not isinstance(self.keras_layer, Activation):
            raise KeyError()

        self.previous = previous

        # mode is set to linear

    def call(self, inputs, **kwargs):

        if self.reuse_slope or self.update_slope:
            extra_kwargs = {"reuse_slope": self.reuse_slope, "update_slope": self.update_slope}
            kwargs.update(extra_kwargs)

        if self.previous:
            return self.call_previous(inputs, **kwargs)
        else:
            return self.call_no_previous(inputs, **kwargs)

    def call_previous(self, inputs, **kwargs):
        output = self.call_no_previous(inputs[:-4], **kwargs)
        return merge_with_previous(output + inputs[-4:])

        # to do

    def call_no_previous(self, inputs, **kwargs):
        return self.activation(inputs, convex_domain=self.convex_domain, mode=self.mode, params=self.params, **kwargs)

    def store_output(self, joint=False, convex_domain=None, reuse_slope=False):
        if convex_domain is None:
            convex_domain = {}
        if joint:
            return not reuse_slope

        return False

    def store_layer(self, joint=False, convex_domain=None, reuse_slope=False):
        if convex_domain is None:
            convex_domain = {}
        if joint:
            return not reuse_slope

        return False


class BackwardNumpyInputLayer(BackwardNumpyLayer):
    def __init__(
        self, keras_layer, previous=False, fusion=True, convex_domain=None, mode=F_FORWARD.name, rec=1, **kwargs
    ):
        """

        :param convex_domain: type of convex input domain (None or dict)
        :param dc_decomp: boolean that indicates whether we return a
        difference of convex decomposition of our layer
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        :param kwargs: extra parameters
        """
        super().__init__(keras_layer=keras_layer, convex_domain=convex_domain, mode=mode, rec=rec, **kwargs)

        if convex_domain is None:
            convex_domain = {}
        if not isinstance(self.keras_layer, InputLayer):
            raise KeyError()

        self.previous = previous
        self.fusion = fusion

    def call_previous(self, inputs):
        if self.fusion:
            return merge_with_previous(inputs[1:])
        else:
            return inputs[-4:]

    def call_no_previous(self, inputs):
        x, _, b_u = inputs[:3]

        shape = np.prod(b_u.shape[1:])
        w_u_ = np.concatenate([np.diag([1] * shape)[None]] * len(x))
        b_u_ = np.zeros_like(b_u)

        return [w_u_, b_u_, w_u_, b_u_]

    def call(self, inputs, **kwargs):
        if self.previous:
            return self.call_previous(inputs)
        else:
            return self.call_no_previous(inputs)


class BackwardNumpyDense(BackwardNumpyLayer):
    def __init__(self, keras_layer, previous=False, convex_domain=None, mode=F_FORWARD.name, rec=1, **kwargs):
        """

        :param convex_domain: type of convex input domain (None or dict)
        :param dc_decomp: boolean that indicates whether we return a
        difference of convex decomposition of our layer
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        :param kwargs: extra parameters
        """
        super().__init__(keras_layer=keras_layer, convex_domain=convex_domain, mode=mode, rec=rec, **kwargs)

        if convex_domain is None:
            convex_domain = {}
        if not isinstance(self.keras_layer, Dense):
            raise KeyError()

        self.activation_name = keras_layer.get_config()["activation"]

        if self.activation_name != "linear":
            raise ValueError("the activation layer should be handled separately from the Dense Layer")

        self.kernel = keras_layer.kernel
        if keras_layer.use_bias:
            self.bias = keras_layer.bias
            self.use_bias = True
        else:
            self.use_bias = False

        self.previous = previous

    def call(self, inputs, **kwargs):
        if self.previous:

            output = self.call_no_previous(inputs)
            return merge_with_previous(output + inputs[-4:])
        else:
            return self.call_no_previous(inputs)

    def call_no_previous(self, inputs):
        x = inputs[0]

        w_ = self.kernel.numpy()[None]
        w_ = np.concatenate([w_] * len(x))

        if self.use_bias:
            b_ = self.bias.numpy()[None]
            b_ = np.concatenate([b_] * len(x))
        else:
            b_ = np.zeros((len(x), self.units))

        return [w_, b_, w_, b_]


def get_backward(keras_layer, previous=True, mode=F_FORWARD.name, convex_domain=None, rec=1, params=None, **kwargs):

    # do it better
    # either a Decomon layer or its pure Keras version
    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    class_name = keras_layer.__class__.__name__

    backward_class_name = f"BackwardNumpy{class_name}"
    class_ = globals()[backward_class_name]
    try:
        return class_(
            keras_layer, previous=previous, mode=mode, convex_domain=convex_domain, rec=rec, params=params, **kwargs
        )
    except KeyError:
        import pdb

        pdb.set_trace()
