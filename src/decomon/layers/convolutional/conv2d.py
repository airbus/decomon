from typing import Optional, Any

from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer, Wrapper
import keras.ops as K
from decomon.constants import Propagation
from decomon.perturbation_domain import BoxDomain, PerturbationDomain
from decomon.layers.convolutional.utils import get_toeplitz_from_layer as get_toeplitz
from decomon.layers.utils import get_bias
from .base_conv import DecomonBaseConv
from decomon.layers.layer import DecomonLayer, DecomonLinearLayer
from typing import Optional
from decomon.types import Tensor

import numpy as np


class DecomonConv2D(DecomonBaseConv):
    layer:Conv2D

        