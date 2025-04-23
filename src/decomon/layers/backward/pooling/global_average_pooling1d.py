from jacobinet.layers.pooling.global_average_pooling1d import (
    BackwardGlobalAveragePooling1D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardGlobalAveragePooling1D(DecomonLinearLayerBackward):
    layer: BackwardGlobalAveragePooling1D
    use_bias = False
    increasing = True
