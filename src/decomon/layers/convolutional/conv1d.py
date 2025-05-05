from typing import Any, Optional

from keras.layers import Conv1D

from .base_conv import DecomonBaseConv


class DecomonConv1D(DecomonBaseConv):
    layer: Conv1D
