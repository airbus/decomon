from jacobinet.layers.reshaping.cropping3d import BackwardCropping3D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardCropping3D(DecomonBackwardLinearLayer):
    layer: BackwardCropping3D
    use_bias = False
    increasing = True
