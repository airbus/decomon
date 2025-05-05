from jacobinet.layers.core.activations import BackwardActivation

from decomon.layers.backward.layer_backward import DecomonNonLinearBackward


class DecomonBackwardActivation(DecomonNonLinearBackward):
    layer: BackwardActivation
