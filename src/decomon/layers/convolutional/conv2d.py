from typing import Any, Optional

from keras.layers import Conv2D  # type:ignore

from .base_conv import DecomonBaseConv


class DecomonConv2D(DecomonBaseConv):
    layer: Conv2D
