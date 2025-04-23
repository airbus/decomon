from jacobinet.layers.reshaping.upsampling2D import BackwardUpSampling2D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardUpSampling2D(DecomonLinearLayerBackward):
    layer: BackwardUpSampling2D
    use_bias = False
    increasing = True
