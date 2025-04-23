from jacobinet.layers.merging.add import BackwardAdd  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearMergeBackward


class DecomonBackwardAdd(DecomonLinearMergeBackward):
    layer: BackwardAdd
    increasing = True
    use_bias = False
