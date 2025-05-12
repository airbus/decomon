from jacobinet.layers.core.activations import BackwardActivation

from decomon.layers.backward.layer_backward import DecomonBackwardNonLinearLayer


class DecomonBackwardActivation(DecomonBackwardNonLinearLayer):
    layer: BackwardActivation
