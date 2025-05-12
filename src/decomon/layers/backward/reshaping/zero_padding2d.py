from jacobinet.layers.reshaping.zero_padding2d import (
    BackwardZeroPadding2D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardZeroPadding2D(DecomonBackwardLinearLayer):
    layer: BackwardZeroPadding2D
    use_bias = False
    increasing = True
