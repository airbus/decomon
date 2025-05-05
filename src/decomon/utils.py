# set global variables
import numpy as np

# memory_limit = int(2e9)
memory_limit: int = 5000000
# if the affine weights of size N, M is such that (N*M)>= memory_limit, we opt for implicit affine propagation or block splitting


def fit_memory(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> bool:
    inner_dim: int = int(np.prod(input_shape) * np.prod(output_shape))
    return inner_dim <= memory_limit
