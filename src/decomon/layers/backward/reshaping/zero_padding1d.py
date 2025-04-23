from jacobinet.layers.reshaping.zero_padding1d import (
    BackwardZeroPadding1D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardZeroPadding1D(DecomonLinearLayerBackward):
    layer: BackwardZeroPadding1D
    use_bias = False
    increasing = True
