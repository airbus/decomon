from jacobinet.layers.reshaping.permute import BackwardPermute  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardPermute(DecomonLinearLayerBackward):
    layer: BackwardPermute
    use_bias = False
    increasing = True
