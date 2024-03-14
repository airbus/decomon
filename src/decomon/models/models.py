from typing import Any, Union

import keras
import keras.ops as K
import numpy as np
from keras import Model
from keras.utils import serialize_keras_object

from decomon.constants import ConvertMethod
from decomon.perturbation_domain import PerturbationDomain


class DecomonModel(keras.Model):
    def __init__(
        self,
        inputs: Union[keras.KerasTensor, list[keras.KerasTensor]],
        outputs: Union[keras.KerasTensor, list[keras.KerasTensor]],
        perturbation_domain: PerturbationDomain,
        method: ConvertMethod,
        ibp: bool,
        affine: bool,
        **kwargs: Any,
    ):
        super().__init__(inputs, outputs, **kwargs)
        self.perturbation_domain = perturbation_domain
        self.method = method
        self.ibp = ibp
        self.affine = affine

    def get_config(self) -> dict[str, Any]:
        # force having functional config which is skipped by default
        # because DecomonModel.__init__() has not same signature as Functional.__init__()
        config = Model(self.inputs, self.outputs).get_config()
        # update with correct name + specific attributes of decomon model
        config.update(
            dict(
                name=self.name,
                perturbation_domain=serialize_keras_object(self.perturbation_domain),
                dc_decomp=self.dc_decomp,
                method=self.method,
                ibp=self.ibp,
                affine=self.affine,
                finetune=self.finetune,
                shared=self.shared,
                backward_bounds=self.backward_bounds,
            )
        )
        return config

    def set_domain(self, perturbation_domain: PerturbationDomain) -> None:
        perturbation_domain = _check_domain(self.perturbation_domain, perturbation_domain)
        self.perturbation_domain = perturbation_domain
        for layer in self.layers:
            if hasattr(layer, "perturbation_domain"):
                layer.perturbation_domain = self.perturbation_domain

    def predict_on_single_batch_np(
        self, inputs: Union[np.ndarray, list[np.ndarray]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Make predictions on numpy arrays fitting in one batch

        Avoid using `self.predict()` known to be not designed for small arrays,
        and leading to memory leaks when used in loops.

        See https://keras.io/api/models/model_training_apis/#predict-method and
        https://github.com/tensorflow/tensorflow/issues/44711

        Args:
            inputs:

        Returns:

        """
        output_tensors = self(inputs)
        if isinstance(output_tensors, list):
            return [K.convert_to_numpy(output) for output in output_tensors]
        else:
            return K.convert_to_numpy(output_tensors)


def _check_domain(
    perturbation_domain_prev: PerturbationDomain, perturbation_domain: PerturbationDomain
) -> PerturbationDomain:
    if type(perturbation_domain) != type(perturbation_domain_prev):
        raise NotImplementedError("We can only change the parameters of the perturbation domain, not its type.")

    return perturbation_domain
