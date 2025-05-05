from typing import Callable, List

import keras.ops as K
from keras.layers import Concatenate, Dense, Input, Reshape
from keras.models import Model, Sequential

from decomon.constants import Propagation


def get_alpha_model_diagonal(
    propagation: Propagation,
    input_shape_wo_batch: List[int],
    model_input_shape: List[int] = [None],
    alpha_model: Callable = None,
):
    if propagation == Propagation.FORWARD:
        return get_alpha_model_diagonal_forward(
            input_shape_wo_batch=input_shape_wo_batch, model_input_shape=model_input_shape, alpha_model=alpha_model
        )
    if propagation == Propagation.BACKWARD:
        return get_alpha_model_diagonal_backward(input_shape_wo_batch=input_shape_wo_batch, alpha_model=alpha_model)
    raise ValueError("unknow propagation mode {}".format(propagation))


def get_alpha_model_diagonal_backward(input_shape_wo_batch: List[int], alpha_model: Callable = None):
    # function description
    # Note: If the input to the layer has a rank greater than 2,
    # Dense computes the dot product between the inputs and the kernel along the last axis of the

    # step 1: create input for lower and upper bounds of the layer
    lower = Input(input_shape_wo_batch)
    upper = Input(input_shape_wo_batch)
    w_lower = Input(input_shape_wo_batch + [None])  # last tensor dimension is unknown
    w_upper = Input(input_shape_wo_batch + [None])  # last tensor dimension is unknown

    # concat everything
    layer_reshape_0 = Reshape(input_shape_wo_batch + [-1, 1])
    layer_reshape_1 = Reshape(input_shape_wo_batch + [-1, 1])
    w_reshaped = [layer_reshape_1(inp) for inp in [w_lower, w_upper]]
    inputs_reshaped = [layer_reshape_0(inp) + 0 * w_reshaped[0] for inp in [lower, upper]]

    inputs_cat = Concatenate(-1)(inputs_reshaped + w_reshaped)
    # apply a sequential layer of your choice
    if alpha_model is None:
        # use a default configuration
        alpha_model = Sequential(
            [
                Dense(100, activation="relu"),
                Dense(80, activation="relu"),
                Dense(20, activation="relu"),
                Dense(1, activation="sigmoid"),
                Reshape(input_shape_wo_batch + [-1]),
            ]
        )

    output = alpha_model(inputs_cat)  # (input_shape_wo_batch+[None])

    # return Model([lower, upper, w_lower, w_upper], K.sum(output, axis=-1))
    return Model([lower, upper, w_lower, w_upper], output)


def get_alpha_model_diagonal_forward(
    input_shape_wo_batch: List[int], model_input_shape: List[int] = [None], alpha_model: Callable = None
):
    # function description
    # Note: If the input to the layer has a rank greater than 2,
    # Dense computes the dot product between the inputs and the kernel along the last axis of the

    # step 1: create input for lower and upper bounds of the layer
    lower = Input(input_shape_wo_batch)
    upper = Input(input_shape_wo_batch)
    w_lower = Input(list(model_input_shape) + input_shape_wo_batch)
    w_upper = Input(list(model_input_shape) + input_shape_wo_batch)
    b_lower = Input(input_shape_wo_batch)
    b_upper = Input(input_shape_wo_batch)

    # concat everything
    layer_reshape_0 = Reshape([-1] + input_shape_wo_batch + [1])
    layer_reshape_1 = Reshape([-1] + input_shape_wo_batch + [1])

    inputs_0 = (
        Concatenate(-1)([layer_reshape_0(inp) for inp in [lower, upper, b_lower, b_upper]])
        + layer_reshape_1(w_lower) * 0
    )
    inputs_1 = Concatenate(-1)([layer_reshape_1(inp) for inp in [w_lower, w_upper]])

    inputs_reshaped = [inputs_0, inputs_1]
    inputs_cat = Concatenate(-1)(inputs_reshaped)

    # apply a sequential layer of your choice
    if alpha_model is None:
        # use a default configuration
        alpha_model = Sequential(
            [
                Dense(100, activation="relu"),
                Dense(80, activation="relu"),
                Dense(20, activation="relu"),
                Dense(1, activation="sigmoid"),
                Reshape([-1] + input_shape_wo_batch),
            ]
        )

    output = alpha_model(inputs_cat)

    return Model([lower, upper, w_lower, b_lower, w_upper, b_upper], output)
