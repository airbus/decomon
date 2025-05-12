from jacobinet.layers.reshaping.zero_padding3d import (
    BackwardZeroPadding3D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardZeroPadding3D(DecomonBackwardLinearLayer):
    layer: BackwardZeroPadding3D
    use_bias = False
    increasing = True
