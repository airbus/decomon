from jacobinet.layers.reshaping.upsampling3d import BackwardUpSampling3D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardUpSampling3D(DecomonLinearLayerBackward):
    layer: BackwardUpSampling3D
    use_bias = False
    increasing = True
