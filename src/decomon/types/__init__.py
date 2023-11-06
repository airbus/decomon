"""Typing module"""


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras

# create extra types for readability


BackendTensor = Any
"""Type for backend tensors.

If the backend is tensorflow, this should be `tf.Tensor`.
For now, there is now exposed way to check wether a tensor is a backend tensor or not (this is a private function in keras)

"""

Tensor = Union[keras.KerasTensor, BackendTensor]
"""Type for any tensor, from keras or backend."""


DecomonInputs = List[Tensor]
