from typing import Any, Dict, List, Optional, Union

import tensorflow as tf

from decomon.layers.core import ForwardMode
from decomon.layers.utils import exp, expand_dims, log, sum
from decomon.utils import add, minus

# compute categorical cross entropy


def categorical_cross_entropy(
    inputs: List[tf.Tensor],
    dc_decomp: bool = False,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    convex_domain: Optional[Dict[str, Any]] = None,
) -> List[tf.Tensor]:

    # step 1: exponential
    if convex_domain is None:
        convex_domain = {}
    outputs = exp(inputs, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp)
    # step 2: sum
    outputs = sum(outputs, axis=-1, mode=mode)
    # step 3: log
    outputs = log(outputs, dc_decomp=dc_decomp, mode=mode, convex_domain=convex_domain)
    outputs = expand_dims(outputs, dc_decomp=dc_decomp, mode=mode, convex_domain=convex_domain, axis=-1)
    # step 4: - inputs
    return add(
        minus(inputs, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp),
        outputs,
        mode=mode,
        convex_domain=convex_domain,
        dc_decomp=dc_decomp,
    )
