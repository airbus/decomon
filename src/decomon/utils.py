import os

import numpy as np

ENV_MEMORY_LIMIT = "DECOMON_MEMORY_LIMIT"


def get_memory_limit() -> int:
    """Get memory limit to be used to decide whether using implicit affine propagation.

     - If environment variable DECOMON_MEMORY_LIMIT is defined we parse it to
      use it as memory limit (it should be something valid for float() or
      int() like "2000", "50_000", or "2e5"),
    - else we use a default value (5_000_000).

    """
    try:
        # we use int(float(...)) to allow strings like "2e9"
        return int(float(os.environ[ENV_MEMORY_LIMIT]))
    except KeyError:
        return 5_000_000


def fit_memory(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> bool:
    """Check whether size(inputs) * size(outputs) <= memory_limit

    memory_limit is given by `get_memory_limit()`
    If returning False, we could for instance opt for implicit affine propagation or block splitting.

    """
    inner_dim: int = int(np.prod(input_shape) * np.prod(output_shape))
    return inner_dim <= get_memory_limit()
