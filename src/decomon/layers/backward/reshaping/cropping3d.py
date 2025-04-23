from jacobinet.layers.reshaping.cropping3d import BackwardCropping3D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardCropping3D(DecomonLinearLayerBackward):
    layer: BackwardCropping3D
    use_bias = False
    increasing = True
