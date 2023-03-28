from typing import Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from decomon.metrics.loss import get_upper_loss
from decomon.models.models import DecomonModel
from decomon.wrapper import get_lower_box, get_upper_box, refine_boxes


#### FORMAL BOUNDS ######
def get_upper_box_tuning(
    model: Union[tf.keras.Model, DecomonModel],
    decomon_model_concat: tf.keras.Model,
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    batch_size: int = 1,
    n_sub_boxes: int = 1,
    lr: float = 0.1,
    epochs: int = 100,
) -> npt.NDArray[np.float_]:
    """upper bound the maximum of a model in a given box

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        batch_size: for computational efficiency, one can split the
            calls to minibatches
        fast: useful in the forward-backward or in the hybrid-backward
            mode to optimize the scores

    Returns:
        numpy array, vector with upper bounds for adversarial attacks
    """

    if np.min(x_max - x_min) < 0:
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    model_ = model

    baseline_upper = get_upper_box(model, x_min, x_max, batch_size=batch_size, n_sub_boxes=n_sub_boxes)

    if n_sub_boxes > 1:
        x_min, x_max = refine_boxes(x_min, x_max, n_sub_boxes)
        # reshape
        shape = list(x_min.shape[2:])
        x_min = np.reshape(x_min, [-1] + shape)
        x_max = np.reshape(x_max, [-1] + shape)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_ = x_min + 0 * x_min
    x_ = x_.reshape([-1] + input_shape)
    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)

    if batch_size > 1:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_min_ = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        X_max_ = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]

        results = [
            get_upper_box_tuning(model_, decomon_model_concat, X_min_[i], X_max_[i], -1, lr=lr, epochs=epochs)
            for i in range(len(X_min_))
        ]

        u_ = np.concatenate(results)

        return u_
    else:

        # freeze_weights
        model.freeze_weights()

        # create loss
        loss_upper = get_upper_loss(model)
        decomon_model_concat.compile(Adam(lr=lr), loss_upper)

        decomon_model_concat.fit(z, baseline_upper, epochs=epochs, verbose=0)

        upper_ = get_upper_box(model, x_min, x_max, batch_size=batch_size, n_sub_boxes=n_sub_boxes)

        # reset alpha
        model.reset_finetuning()

        return np.minimum(baseline_upper, upper_)


#### FORMAL BOUNDS ######
def get_lower_box_tuning(
    model: Union[tf.keras.Model, DecomonModel],
    decomon_model_concat: tf.keras.Model,
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    batch_size: int = 1,
    n_sub_boxes: int = 1,
    lr: float = 0.1,
    epochs: int = 100,
) -> npt.NDArray[np.float_]:
    """upper bound the maximum of a model in a given box

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        batch_size: for computational efficiency, one can split the
            calls to minibatches
        fast: useful in the forward-backward or in the hybrid-backward
            mode to optimize the scores

    Returns:
        numpy array, vector with upper bounds for adversarial attacks
    """

    if np.min(x_max - x_min) < 0:
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    model_ = model

    baseline_upper = get_lower_box(model, x_min, x_max, batch_size=batch_size, n_sub_boxes=n_sub_boxes)

    if n_sub_boxes > 1:
        x_min, x_max = refine_boxes(x_min, x_max, n_sub_boxes)
        # reshape
        shape = list(x_min.shape[2:])
        x_min = np.reshape(x_min, [-1] + shape)
        x_max = np.reshape(x_max, [-1] + shape)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_ = x_min + 0 * x_min
    x_ = x_.reshape([-1] + input_shape)
    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)

    if batch_size > 0 and batch_size != len(x_min):
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_min_ = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        X_max_ = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]

        results = [
            get_lower_box_tuning(model_, decomon_model_concat, X_min_[i], X_max_[i], -1, lr=lr, epochs=epochs)
            for i in range(len(X_min_))
        ]

        u_ = np.concatenate(results)
        model.reset_finetuning()

        return u_
    else:

        # freeze_weights
        model.freeze_weights()

        # create loss
        loss_upper = get_upper_loss(model)
        decomon_model_concat.compile(Adam(lr=lr), loss_upper)

        decomon_model_concat.fit(z, baseline_upper, epochs=epochs, verbose=0)

        upper_ = get_lower_box(model, x_min, x_max, batch_size=batch_size, n_sub_boxes=n_sub_boxes)

        # reset alpha
        model.reset_finetuning()

        return np.minimum(baseline_upper, upper_)
