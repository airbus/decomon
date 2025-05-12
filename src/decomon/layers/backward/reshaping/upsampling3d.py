from jacobinet.layers.reshaping.upsampling3d import BackwardUpSampling3D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardUpSampling3D(DecomonBackwardLinearLayer):
    layer: BackwardUpSampling3D
    use_bias = False
    increasing = True
