from jacobinet.layers.reshaping.repeat_vector import (
    BackwardRepeatVector,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardRepeatVector(DecomonLinearLayerBackward):
    layer: BackwardRepeatVector
    use_bias = False
    increasing = True
