from jacobinet.layers.reshaping.upsampling1d import BackwardUpSampling1D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardUpSampling1D(DecomonBackwardLinearLayer):
    layer: BackwardUpSampling1D
    use_bias = False
    increasing = True
