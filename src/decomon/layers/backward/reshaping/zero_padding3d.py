from jacobinet.layers.reshaping.zero_padding3d import (
    BackwardZeroPadding3D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardZeroPadding3D(DecomonLinearLayerBackward):
    layer: BackwardZeroPadding3D
    use_bias = False
    increasing = True
