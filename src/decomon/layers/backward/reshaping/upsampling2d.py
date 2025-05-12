from jacobinet.layers.reshaping.upsampling2D import BackwardUpSampling2D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardUpSampling2D(DecomonBackwardLinearLayer):
    layer: BackwardUpSampling2D
    use_bias = False
    increasing = True
