from jacobinet.layers.reshaping.permute import BackwardPermute  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardPermute(DecomonBackwardLinearLayer):
    layer: BackwardPermute
    use_bias = False
    increasing = True
