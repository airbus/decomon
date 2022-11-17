from __future__ import absolute_import

from abc import ABC, abstractmethod

from tensorflow.keras.layers import Layer

from decomon.layers.core import F_HYBRID


class GradientLayer(ABC, Layer):
    """
    Abstract class that contains the common information of every implemented layers
    """

    def __init__(
        self, convex_domain=None, mode=F_HYBRID.name, input_mode=F_HYBRID.name, finetune=False, shared=False, **kwargs
    ):
        if convex_domain is None:
            convex_domain = {}
        self.convex_domain = convex_domain
        self.mode = mode
        self.input_mode = input_mode
