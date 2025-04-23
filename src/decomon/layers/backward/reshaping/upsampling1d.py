from jacobinet.layers.reshaping.upsampling1d import BackwardUpSampling1D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardUpSampling1D(DecomonLinearLayerBackward):
    layer: BackwardUpSampling1D
    use_bias = False
    increasing = True
