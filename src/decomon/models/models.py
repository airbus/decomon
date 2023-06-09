import copy
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from keras.engine.functional import get_network_config

from decomon.layers.core import StaticVariables
from decomon.models.utils import ConvertMethod
from decomon.utils import ConvexDomainType, Option


class DecomonModel(tf.keras.Model):
    def __init__(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        outputs: Union[tf.Tensor, List[tf.Tensor]],
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        method: Union[str, ConvertMethod] = ConvertMethod.FORWARD_AFFINE,
        ibp: bool = True,
        affine: bool = True,
        finetune: bool = False,
        shared: bool = True,
        backward_bounds: bool = False,
        **kwargs: Any,
    ):
        super().__init__(inputs, outputs, **kwargs)
        if convex_domain is None:
            convex_domain = {}
        self.convex_domain = convex_domain
        self.nb_tensors = StaticVariables(dc_decomp).nb_tensors
        self.dc_decomp = dc_decomp
        self.method = ConvertMethod(method)
        self.ibp = ibp
        self.affine = affine
        self.finetune = finetune
        self.backward_bounds = backward_bounds
        self.shared = shared

    def set_domain(self, convex_domain: Dict[str, Any]) -> None:
        convex_domain = _check_domain(self.convex_domain, convex_domain)
        self.convex_domain = convex_domain
        for layer in self.layers:
            if hasattr(layer, "convex_domain"):
                layer.convex_domain = self.convex_domain

    def freeze_weights(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "freeze_weights"):
                layer.freeze_weights()

    def unfreeze_weights(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "unfreeze_weights"):
                layer.unfreeze_weights()

    def freeze_alpha(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "freeze_alpha"):
                layer.freeze_alpha()

    def unfreeze_alpha(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "unfreeze_alpha"):
                layer.unfreeze_alpha()

    def reset_finetuning(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_finetuning"):
                layer.reset_finetuning()

    def get_config(self) -> Dict[str, Any]:
        # Continue adding configs into what the super class has added.
        config = super().get_config()
        return copy.deepcopy(get_network_config(self, config=config))


def _check_domain(convex_domain_prev: Dict[str, Any], convex_domain: Dict[str, Any]) -> Dict[str, Any]:
    if len(convex_domain) == 0:
        convex_domain_type = ConvexDomainType.BOX
    else:
        convex_domain_type = ConvexDomainType(convex_domain["name"])

    if len(convex_domain_prev) == 0:
        convex_domain_prev_type = ConvexDomainType.BOX
    else:
        convex_domain_prev_type = ConvexDomainType(convex_domain_prev["name"])

    if convex_domain_prev_type != convex_domain_type:
        raise NotImplementedError("We can only change the parameters of the convex domain, not its nature.")

    return convex_domain


def get_AB(model: DecomonModel) -> Dict[str, List[tf.Variable]]:
    dico_AB: Dict[str, List[tf.Variable]] = {}
    convex_domain = model.convex_domain
    if not (
        len(convex_domain)
        and ConvexDomainType(convex_domain["name"]) == ConvexDomainType.GRID
        and Option(convex_domain["option"]) == Option.milp
    ):
        return dico_AB

    for layer in model.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.grid_finetune
    return dico_AB


def get_AB_finetune(model: DecomonModel) -> Dict[str, tf.Variable]:
    dico_AB: Dict[str, tf.Variable] = {}
    convex_domain = model.convex_domain
    if not (
        len(convex_domain)
        and ConvexDomainType(convex_domain["name"]) == ConvexDomainType.GRID
        and Option(convex_domain["option"]) == Option.milp
    ):
        return dico_AB

    if not model.finetune:
        return dico_AB

    for layer in model.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.alpha_b_l
    return dico_AB
