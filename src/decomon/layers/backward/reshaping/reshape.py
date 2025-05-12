from jacobinet.layers.reshaping.reshape import BackwardReshape  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardReshape(DecomonBackwardLinearLayer):
    layer: BackwardReshape
    use_bias = False
    increasing = True
