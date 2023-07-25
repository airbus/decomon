from typing import Any, Dict, List, Optional, Union

import tensorflow as tf

from decomon.core import BoxDomain, ForwardMode, PerturbationDomain
from decomon.layers.utils import exp, expand_dims, log, sum
from decomon.utils import add, minus

# compute categorical cross entropy


def categorical_cross_entropy(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    perturbation_domain: Optional[PerturbationDomain] = None,
) -> List[tf.Tensor]:
    # step 1: exponential
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    outputs = exp(inputs, mode=mode, perturbation_domain=perturbation_domain, dc_decomp=dc_decomp)
    # step 2: sum
    outputs = sum(outputs, axis=-1, mode=mode)
    # step 3: log
    outputs = log(outputs, dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    outputs = expand_dims(outputs, dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain, axis=-1)
    # step 4: - inputs
    return add(
        minus(inputs, mode=mode, perturbation_domain=perturbation_domain, dc_decomp=dc_decomp),
        outputs,
        mode=mode,
        perturbation_domain=perturbation_domain,
        dc_decomp=dc_decomp,
    )
