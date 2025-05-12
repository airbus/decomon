from jacobinet.layers.pooling.average_pooling1d import (
    BackwardAveragePooling1D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardAveragePooling1D(DecomonBackwardLinearLayer):
    layer: BackwardAveragePooling1D
    use_bias = False
    increasing = True
