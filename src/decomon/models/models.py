import copy
from typing import Dict, List

import tensorflow as tf
from keras.engine.functional import get_network_config

from decomon.layers.core import StaticVariables
from decomon.models.utils import ConvertMethod
from decomon.utils import ConvexDomainType


class DecomonModel(tf.keras.Model):
    def __init__(
        self,
        input,
        output,
        convex_domain=None,
        dc_decomp=False,
        method=ConvertMethod.FORWARD_AFFINE,
        optimize="True",
        IBP=True,
        forward=True,
        finetune=False,
        shared=True,
        backward_bounds=False,
        **kwargs,
    ):
        super().__init__(input, output, **kwargs)
        if convex_domain is None:
            convex_domain = {}
        self.convex_domain = convex_domain
        self.optimize = optimize
        self.nb_tensors = StaticVariables(dc_decomp).nb_tensors
        self.dc_decomp = dc_decomp
        self.method = ConvertMethod(method)
        self.IBP = IBP
        self.forward = forward
        self.finetune = finetune
        self.backward_bounds = backward_bounds

    def set_domain(self, convex_domain):
        convex_domain = set_domain_priv(self.convex_domain, convex_domain)
        self.convex_domain = convex_domain
        for layer in self.layers:
            if hasattr(layer, "convex_domain"):
                layer.convex_domain = self.convex_domain

    def freeze_weights(self):
        for layer in self.layers:
            if hasattr(layer, "freeze_weights"):
                layer.freeze_weights()

    def unfreeze_weights(self):
        for layer in self.layers:
            if hasattr(layer, "unfreeze_weights"):
                layer.unfreeze_weights()

    def freeze_alpha(self):
        for layer in self.layers:
            if hasattr(layer, "freeze_alpha"):
                layer.freeze_alpha()

    def unfreeze_alpha(self):
        for layer in self.layers:
            if hasattr(layer, "unfreeze_alpha"):
                layer.unfreeze_alpha()

    def reset_finetuning(self):
        for layer in self.layers:
            if hasattr(layer, "reset_finetuning"):
                layer.reset_finetuning()

    def get_config(self):
        # Continue adding configs into what the super class has added.
        config = super().get_config()
        return copy.deepcopy(get_network_config(self, config=config))


def set_domain_priv(convex_domain_prev, convex_domain):
    msg = "we can only change the parameters of the convex domain, not its nature"

    convex_domain_ = convex_domain
    if convex_domain == {}:
        convex_domain = {"name": ConvexDomainType.BOX}

    if len(convex_domain_prev) == 0 or convex_domain_prev["name"] == ConvexDomainType.BOX:
        # Box
        if convex_domain["name"] != ConvexDomainType.BOX:
            raise NotImplementedError(msg)

    if convex_domain_prev["name"] != convex_domain["name"]:
        raise NotImplementedError(msg)

    return convex_domain_


def get_AB(model_: DecomonModel) -> Dict[str, List[tf.Variable]]:
    dico_AB: Dict[str, List[tf.Variable]] = {}
    convex_domain = model_.convex_domain
    if not (
        len(convex_domain) and convex_domain["name"] == ConvexDomainType.GRID and convex_domain["option"] == "milp"
    ):
        return dico_AB

    for layer in model_.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.grid_finetune
    return dico_AB


def get_AB_finetune(model_: DecomonModel) -> Dict[str, tf.Variable]:
    dico_AB: Dict[str, tf.Variable] = {}
    convex_domain = model_.convex_domain
    if not (
        len(convex_domain) and convex_domain["name"] == ConvexDomainType.GRID and convex_domain["option"] == "milp"
    ):
        return dico_AB

    if not model_.finetune:
        return dico_AB

    for layer in model_.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.alpha_b_l
    return dico_AB
