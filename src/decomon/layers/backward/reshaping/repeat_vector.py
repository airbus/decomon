from jacobinet.layers.reshaping.repeat_vector import (
    BackwardRepeatVector,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardRepeatVector(DecomonBackwardLinearLayer):
    layer: BackwardRepeatVector
    use_bias = False
    increasing = True
