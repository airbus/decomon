from jacobinet.layers.reshaping.flatten import BackwardFlatten  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardFlatten(DecomonBackwardLinearLayer):
    layer: BackwardFlatten
    use_bias = False
    increasing = True
