from jacobinet.layers.reshaping.cropping2d import BackwardCropping2D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardCropping2D(DecomonBackwardLinearLayer):
    layer: BackwardCropping2D
    use_bias = False
    increasing = True
