from typing import Any, Dict, List, Optional, Union

import keras

from decomon.core import (
    BoxDomain,
    GridDomain,
    InputsOutputsSpec,
    Option,
    PerturbationDomain,
)
from decomon.models.utils import ConvertMethod


class DecomonModel(keras.Model):
    def __init__(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        outputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        perturbation_domain: Optional[PerturbationDomain] = None,
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
        if perturbation_domain is None:
            perturbation_domain = BoxDomain()
        self.perturbation_domain = perturbation_domain
        self.dc_decomp = dc_decomp
        self.method = ConvertMethod(method)
        self.ibp = ibp
        self.affine = affine
        self.finetune = finetune
        self.backward_bounds = backward_bounds
        self.shared = shared

    def set_domain(self, perturbation_domain: PerturbationDomain) -> None:
        perturbation_domain = _check_domain(self.perturbation_domain, perturbation_domain)
        self.perturbation_domain = perturbation_domain
        for layer in self.layers:
            if hasattr(layer, "perturbation_domain"):
                layer.perturbation_domain = self.perturbation_domain

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


def _check_domain(
    perturbation_domain_prev: PerturbationDomain, perturbation_domain: PerturbationDomain
) -> PerturbationDomain:
    if type(perturbation_domain) != type(perturbation_domain_prev):
        raise NotImplementedError("We can only change the parameters of the perturbation domain, not its type.")

    return perturbation_domain


def get_AB(model: DecomonModel) -> Dict[str, List[keras.Variable]]:
    dico_AB: Dict[str, List[keras.Variable]] = {}
    perturbation_domain = model.perturbation_domain
    if not (isinstance(perturbation_domain, GridDomain) and perturbation_domain.opt_option == Option.milp):
        return dico_AB

    for layer in model.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.grid_finetune
    return dico_AB


def get_AB_finetune(model: DecomonModel) -> Dict[str, keras.Variable]:
    dico_AB: Dict[str, keras.Variable] = {}
    perturbation_domain = model.perturbation_domain
    if not (isinstance(perturbation_domain, GridDomain) and perturbation_domain.opt_option == Option.milp):
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
