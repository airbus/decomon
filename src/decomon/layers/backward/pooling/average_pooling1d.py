from jacobinet.layers.pooling.average_pooling1d import (
    BackwardAveragePooling1D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardAveragePooling1D(DecomonLinearLayerBackward):
    layer: BackwardAveragePooling1D
    use_bias = False
    increasing = True
