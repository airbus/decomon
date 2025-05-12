from jacobinet.layers.pooling.global_average_pooling2d import (
    BackwardGlobalAveragePooling2D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardGlobalAveragePooling2D(DecomonBackwardLinearLayer):
    layer: BackwardGlobalAveragePooling2D
    use_bias = False
    increasing = True
