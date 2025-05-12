from jacobinet.layers.pooling.average_pooling2d import (
    BackwardAveragePooling2D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardAveragePooling2D(DecomonBackwardLinearLayer):
    layer: BackwardAveragePooling2D
    use_bias = False
    increasing = True
