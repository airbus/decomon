from jacobinet.layers.reshaping.cropping2d import BackwardCropping2D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardCropping2D(DecomonLinearLayerBackward):
    layer: BackwardCropping2D
    use_bias = False
    increasing = True
