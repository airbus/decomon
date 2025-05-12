from jacobinet.layers.reshaping.zero_padding1d import (
    BackwardZeroPadding1D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardZeroPadding1D(DecomonBackwardLinearLayer):
    layer: BackwardZeroPadding1D
    use_bias = False
    increasing = True
