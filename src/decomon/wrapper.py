from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from decomon.core import BallDomain, BoxDomain, GridDomain
from decomon.models.convert import clone
from decomon.models.models import DecomonModel
from decomon.models.utils import ConvertMethod

IntegerType = Union[int, np.int_]
"""Alias for integers types."""
LabelType = Union[IntegerType, Sequence[IntegerType], np.typing.NDArray[np.int_]]
"""Alias for labels types."""


def _get_dual_ord(p: float) -> float:
    if p == np.inf:
        return 1
    elif p == 1:
        return np.inf
    elif p == 2:
        return 2
    else:
        raise ValueError(f"p must be equal to 1, 2, or np.inf, unknown value {p}.")


def prepare_labels(labels: LabelType, n_batch: int) -> np.typing.NDArray[np.int_]:
    if isinstance(labels, (int, np.int_)):
        labels = np.zeros((n_batch, 1), dtype=np.int_) + labels
    elif not isinstance(labels, np.ndarray):
        labels = np.array(labels).reshape((n_batch, -1))
    else:
        labels = labels.reshape((n_batch, -1))
    return labels.astype("int_")


##### ADVERSARIAL ROBUSTTNESS #####
def get_adv_box(
    model: Union[keras.Model, DecomonModel],
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    source_labels: LabelType,
    target_labels: Optional[LabelType] = None,
    batch_size: int = -1,
    n_sub_boxes: int = 1,
) -> npt.NDArray[np.float_]:
    """if the output is negative, then it is a formal guarantee that there is no adversarial examples

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        source_labels: the list of label that should be predicted all
            the time in the box (either an integer, either an array that
            can contain multiple source labels for each sample)
        target_labels: the list of label that should never be predicted
            in the box (either an integer, either an array that can
            contain multiple target labels for each sample)
        batch_size: for computational efficiency, one can split the
            calls to minibatches
        n_sub_boxes:
    Returns:
        numpy array, vector with upper bounds for adversarial attacks
    """
    if np.min(x_max - x_min) < 0:
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    decomon_model: DecomonModel
    if not isinstance(model, DecomonModel):
        decomon_model = clone(model)
    else:
        assert isinstance(model.perturbation_domain, (BoxDomain, GridDomain))
        decomon_model = model

    n_split = 1
    n_batch = len(x_min)

    if n_sub_boxes > 1:
        x_min, x_max = refine_boxes(x_min, x_max, n_sub_boxes)
        # reshape
        n_split = x_min.shape[1]
        shape = list(x_min.shape[2:])
        x_min = np.reshape(x_min, [-1] + shape)
        x_max = np.reshape(x_max, [-1] + shape)

    # reshape x_mmin, x_max
    if decomon_model.backward_bounds:
        input_shape = list(decomon_model.input_shape[0][2:])
    else:
        input_shape = list(decomon_model.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_reshaped = x_min + 0 * x_min
    x_reshaped = x_reshaped.reshape([-1] + input_shape)

    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)

    source_labels = prepare_labels(source_labels, n_batch)
    if target_labels is not None:
        target_labels = prepare_labels(target_labels, n_batch)

    if n_split > 1:
        shape = list(source_labels.shape[1:])
        source_labels = np.reshape(np.concatenate([source_labels[:, None]] * n_split, 1), [-1] + shape)
        if target_labels is not None:
            target_labels = np.reshape(np.concatenate([target_labels[:, None]] * n_split, 1), [-1] + shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_reshaped) % batch_size > 0:
            r += 1
        x_min_list = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        x_max_list = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        source_labels_list = [
            source_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)
        ]
        if target_labels is not None:
            target_labels_list = [
                target_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)
            ]
            adv_score = np.concatenate(
                [
                    get_adv_box(
                        decomon_model, x_min_list[i], x_max_list[i], source_labels_list[i], target_labels_list[i], -1
                    )
                    for i in range(len(x_min_list))
                ]
            )
        else:
            adv_score = np.concatenate(
                [
                    get_adv_box(decomon_model, x_min_list[i], x_max_list[i], source_labels_list[i], None, -1)
                    for i in range(len(x_min_list))
                ]
            )

    else:
        ibp = decomon_model.ibp
        affine = decomon_model.affine
        n_label = source_labels.shape[-1]

        # two possitible cases: the model improves the bound based on the knowledge of the labels
        output: npt.NDArray[np.float_]
        if decomon_model.backward_bounds:
            C = np.diag([1] * n_label)[None] - source_labels[:, :, None]
            output = decomon_model.predict([z, C], verbose=0)
        else:
            output = decomon_model.predict(z, verbose=0)

        def get_ibp_score(
            u_c: npt.NDArray[np.float_],
            l_c: npt.NDArray[np.float_],
            source_tensor: npt.NDArray[np.int_],
            target_tensor: Optional[npt.NDArray[np.int_]] = None,
            backward: bool = False,
        ) -> npt.NDArray[np.float_]:
            if target_tensor is None:
                target_tensor = 1 - source_tensor

            shape = np.prod(u_c.shape[1:])

            t_tensor_reshaped = np.reshape(target_tensor, (-1, shape))
            s_tensor_reshaped = np.reshape(source_tensor, (-1, shape))

            # add penalties on biases
            if backward:
                return np.max(u_c, -1)
            else:
                upper = u_c[:, :, None] - l_c[:, None]
            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_reshaped[:, :, None] * s_tensor_reshaped[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        def get_affine_score(
            z_tensor: npt.NDArray[np.float_],
            w_u: npt.NDArray[np.float_],
            b_u: npt.NDArray[np.float_],
            w_l: npt.NDArray[np.float_],
            b_l: npt.NDArray[np.float_],
            source_tensor: npt.NDArray[np.int_],
            target_tensor: Optional[npt.NDArray[np.int_]] = None,
            backward: bool = False,
        ) -> npt.NDArray[np.float_]:
            if target_tensor is None:
                target_tensor = 1 - source_tensor

            n_dim = w_u.shape[1]
            shape = np.prod(b_u.shape[1:])
            w_u_reshaped = np.reshape(w_u, (-1, n_dim, shape, 1))
            w_l_reshaped = np.reshape(w_l, (-1, n_dim, 1, shape))
            b_u_reshaped = np.reshape(b_u, (-1, shape, 1))
            b_l_reshaped = np.reshape(b_l, (-1, 1, shape))

            t_tensor_reshaped = np.reshape(target_tensor, (-1, shape))
            s_tensor_reshaped = np.reshape(source_tensor, (-1, shape))

            if not backward:
                w_u_f = w_u_reshaped - w_l_reshaped
                b_u_f = b_u_reshaped - b_l_reshaped
            else:
                upper = (
                    (np.maximum(w_u, 0.0) * z_tensor[:, 1][:, :, None]).sum(1)
                    + (np.minimum(w_u, 0.0) * z_tensor[:, 0][:, :, None]).sum(1)
                    + b_u
                )

                return upper.max(-1)

            # add penalties on biases
            upper = (
                np.sum(np.maximum(w_u_f, 0) * z_tensor[:, 1, :, None, None], 1)
                + np.sum(np.minimum(w_u_f, 0) * z_tensor[:, 0, :, None, None], 1)
                + b_u_f
            )  # (-1, shape, shape)
            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_reshaped[:, :, None] * s_tensor_reshaped[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        if ibp and affine:
            z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:7]
        elif not ibp and affine:
            z, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]
        elif ibp and not affine:
            u_c, l_c = output[:2]
        else:
            raise NotImplementedError("not ibp and not affine not implemented")

        if ibp:
            adv_ibp = get_ibp_score(u_c, l_c, source_labels, target_labels, backward=decomon_model.backward_bounds)
        if affine:
            adv_f = get_affine_score(
                z, w_u_f, b_u_f, w_l_f, b_l_f, source_labels, target_labels, backward=decomon_model.backward_bounds
            )

        if ibp and not affine:
            adv_score = adv_ibp
        elif ibp and affine:
            adv_score = np.minimum(adv_ibp, adv_f)
        elif not ibp and affine:
            adv_score = adv_f
        else:
            raise NotImplementedError("not ibp and not affine not implemented")

    if n_split > 1:
        adv_score = np.max(np.reshape(adv_score, (-1, n_split)), -1)

    return adv_score


def check_adv_box(
    model: Union[keras.Model, DecomonModel],
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    source_labels: npt.NDArray[np.int_],
    target_labels: Optional[npt.NDArray[np.int_]] = None,
    batch_size: int = -1,
) -> npt.NDArray[np.float_]:
    """if the output is negative, then it is a formal guarantee that there is no adversarial examples

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        source_labels: the list of label that should be predicted all
            the time in the box (either an integer, either an array that
            can contain multiple source labels for each sample)
        target_labels: the list of label that should never be predicted
            in the box (either an integer, either an array that can
            contain multiple target labels for each sample)
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        numpy array, vector with upper bounds for adversarial attacks
    """
    if np.min(x_max - x_min) < 0:
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    if not isinstance(model, DecomonModel):
        decomon_model = clone(model)
    else:
        assert isinstance(model.perturbation_domain, (BoxDomain, GridDomain))
        decomon_model = model

    ibp = decomon_model.ibp
    affine = decomon_model.affine

    n_split = 1
    n_batch = len(x_min)
    # reshape x_mmin, x_max
    input_shape = list(decomon_model.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_reshaped = x_min + 0 * x_min
    x_reshaped = x_reshaped.reshape([-1] + input_shape)

    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)
    if isinstance(source_labels, (int, np.int_)):
        source_labels = np.zeros((n_batch, 1)) + source_labels

    if isinstance(source_labels, list):
        source_labels = np.array(source_labels).reshape((n_batch, -1))

    source_labels = source_labels.reshape((n_batch, -1))
    source_labels = source_labels.astype(np.int_)

    if target_labels is not None:
        target_labels = target_labels.reshape((n_batch, -1))
        target_labels = target_labels.astype(np.int_)

    if n_split > 1:
        shape = list(source_labels.shape[1:])
        source_labels = np.reshape(np.concatenate([source_labels[:, None]] * n_split, 1), [-1] + shape)
        if target_labels is not None:
            target_labels = np.reshape(np.concatenate([target_labels[:, None]] * n_split, 1), [-1] + shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_reshaped) % batch_size > 0:
            r += 1
        x_min_list = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        x_max_list = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        source_labels_list = [
            source_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)
        ]
        target_labels_list: List[Optional[npt.NDArray[np.int_]]]
        if (
            (target_labels is not None)
            and (not isinstance(target_labels, int))
            and (str(target_labels.dtype)[:3] != "int")
        ):
            target_labels_list = [
                target_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)
            ]
        else:
            target_labels_list = [target_labels] * (len(x_reshaped) // batch_size + r)

        return np.concatenate(
            [
                check_adv_box(
                    decomon_model, x_min_list[i], x_max_list[i], source_labels_list[i], target_labels_list[i], -1
                )
                for i in range(len(x_min_list))
            ]
        )

    else:
        output = decomon_model.predict(z, verbose=0)

        if not affine:
            # translate  into affine information
            u_c = output[0]
            w_u = 0 * u_c[:, None] + np.zeros((1, input_dim, 1))
            output = [z, w_u, u_c, w_u, output[-1]]
            ibp = False
            affine = True

        def get_affine_sample(
            z_tensor: npt.NDArray[np.float_],
            w_u: npt.NDArray[np.float_],
            b_u: npt.NDArray[np.float_],
            w_l: npt.NDArray[np.float_],
            b_l: npt.NDArray[np.float_],
            source_tensor: npt.NDArray[np.int_],
            target_tensor: Optional[npt.NDArray[np.int_]] = None,
        ) -> npt.NDArray[np.float_]:
            if target_tensor is None:
                target_tensor = 1 - source_tensor

            n_dim = w_u.shape[1]
            shape = np.prod(b_u.shape[1:])
            w_u_reshaped = np.reshape(w_u, (-1, n_dim, shape, 1))
            w_l_reshaped = np.reshape(w_l, (-1, n_dim, 1, shape))
            b_u_reshaped = np.reshape(b_u, (-1, shape, 1))
            b_l_reshaped = np.reshape(b_l, (-1, 1, shape))

            t_tensor_reshaped = np.reshape(target_tensor, (-1, shape))
            s_tensor_reshaped = np.reshape(source_tensor, (-1, shape))

            w_u_f = w_u_reshaped - w_l_reshaped
            b_u_f = b_u_reshaped - b_l_reshaped

            # add penalties on biases
            upper = (
                np.sum(np.maximum(w_u_f, 0) * z_tensor[:, 1, :, None, None], 1)
                + np.sum(np.minimum(w_u_f, 0) * z_tensor[:, 0, :, None, None], 1)
                + b_u_f
            )  # (-1, shape, shape)
            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_reshaped[:, :, None] * s_tensor_reshaped[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        if ibp:
            z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:7]
        else:
            z, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]

        return get_affine_sample(z, w_l_f, b_l_f, w_u_f, b_u_f, source_labels)


#### FORMAL BOUNDS ######
def get_upper_box(
    model: Union[keras.Model, DecomonModel],
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    batch_size: int = -1,
    n_sub_boxes: int = 1,
) -> npt.NDArray[np.float_]:
    """upper bound the maximum of a model in a given box

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        numpy array, vector with upper bounds for adversarial attacks
    """

    return get_range_box(model=model, x_min=x_min, x_max=x_max, batch_size=batch_size, n_sub_boxes=n_sub_boxes)[0]


def get_lower_box(
    model: Union[keras.Model, DecomonModel],
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    batch_size: int = -1,
    n_sub_boxes: int = 1,
) -> npt.NDArray[np.float_]:
    """lower bound the minimum of a model in a given box

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        numpy array, vector with lower bounds for adversarial attacks
    """

    return get_range_box(model=model, x_min=x_min, x_max=x_max, batch_size=batch_size, n_sub_boxes=n_sub_boxes)[1]


def get_range_box(
    model: Union[keras.Model, DecomonModel],
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    batch_size: int = -1,
    n_sub_boxes: int = 1,
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """bounding the outputs of a model in a given box
    if the constant is negative, then it is a formal guarantee that there is no adversarial examples

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        2 numpy array, vector with upper bounds and vector with lower
        bounds
    """
    if np.min(x_max - x_min) < 0:
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    if not (isinstance(model, DecomonModel)):
        decomon_model = clone(model)
    else:
        assert isinstance(model.perturbation_domain, (BoxDomain, GridDomain))
        decomon_model = model

    n_split = 1
    n_batch = len(x_min)

    if n_sub_boxes > 1:
        x_min, x_max = refine_boxes(x_min, x_max, n_sub_boxes)
        # reshape
        n_split = x_min.shape[1]
        shape = list(x_min.shape[2:])
        x_min = np.reshape(x_min, [-1] + shape)
        x_max = np.reshape(x_max, [-1] + shape)

    # reshape x_mmin, x_max
    input_shape = list(decomon_model.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_reshaped = x_min + 0 * x_min
    x_reshaped = x_reshaped.reshape([-1] + input_shape)
    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)

    if batch_size > 0:
        # split
        r = 0
        if len(x_reshaped) % batch_size > 0:
            r += 1
        x_min_list = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        x_max_list = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        results = [get_range_box(decomon_model, x_min_list[i], x_max_list[i], -1) for i in range(len(x_min_list))]

        u_out = np.concatenate([r[0] for r in results])
        l_out = np.concatenate([r[1] for r in results])

    else:
        ibp = decomon_model.ibp
        affine = decomon_model.affine

        output = decomon_model.predict(z, verbose=0)
        shape = list(output[-1].shape[1:])
        output_dim = np.prod(shape)

        if affine:
            if ibp:
                _, u_i, w_u_f, b_u_f, l_i, w_l_f, b_l_f = output[:7]
                if len(u_i.shape) > 2:
                    u_i = np.reshape(u_i, (-1, output_dim))
                    l_i = np.reshape(l_i, (-1, output_dim))
            else:
                _, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]

            # reshape if necessary
            if len(w_u_f.shape) > 3:
                w_u_f = np.reshape(w_u_f, (-1, input_dim, output_dim))
                w_l_f = np.reshape(w_l_f, (-1, input_dim, output_dim))
                b_u_f = np.reshape(b_u_f, (-1, output_dim))
                b_l_f = np.reshape(b_l_f, (-1, output_dim))

            u_f = (
                np.sum(np.maximum(w_u_f, 0) * x_max[:, 0, :, None], 1)
                + np.sum(np.minimum(w_u_f, 0) * x_min[:, 0, :, None], 1)
                + b_u_f
            )
            l_f = (
                np.sum(np.maximum(w_l_f, 0) * x_min[:, 0, :, None], 1)
                + np.sum(np.minimum(w_l_f, 0) * x_max[:, 0, :, None], 1)
                + b_l_f
            )

        else:
            u_i = output[0]
            l_i = output[1]
            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (-1, output_dim))
                l_i = np.reshape(l_i, (-1, output_dim))

        if ibp and affine:
            u_out = np.minimum(u_i, u_f)
            l_out = np.maximum(l_i, l_f)
        elif ibp and not affine:
            u_out = u_i
            l_out = l_i
        elif not ibp and affine:
            u_out = u_f
            l_out = l_f
        else:
            raise NotImplementedError("not ibp and not affine not implemented")

        #####
        if len(shape) > 1:
            u_out = np.reshape(u_out, [-1] + shape)
            l_out = np.reshape(l_out, [-1] + shape)

    if n_split > 1:
        u_out = np.max(np.reshape(u_out, (-1, n_split)), -1)
        l_out = np.min(np.reshape(l_out, (-1, n_split)), -1)
    return u_out, l_out


# get upper bound of a sample with bounded noise
def get_upper_noise(
    model: Union[keras.Model, DecomonModel],
    x: npt.NDArray[np.float_],
    eps: float,
    p: float = np.inf,
    batch_size: int = -1,
) -> npt.NDArray[np.float_]:
    """upper bound the maximum of a model in an Lp Ball

    Args:
        model: either a Keras model or a Decomon model
        x: numpy array, the example around
        eps: the radius of the ball
        p: the type of Lp norm (p=2, 1, np.inf)
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        numpy array, vector with upper bounds of the range of values
        taken by the model inside the ball
    """

    return get_range_noise(model=model, x=x, eps=eps, p=p, batch_size=batch_size)[0]


# get upper bound of a sample with bounded noise
def get_lower_noise(
    model: Union[keras.Model, DecomonModel],
    x: npt.NDArray[np.float_],
    eps: float,
    p: float = np.inf,
    batch_size: int = -1,
) -> npt.NDArray[np.float_]:
    """lower bound the minimum of a model in an Lp Ball

    Args:
        model: either a Keras model or a Decomon model
        x: numpy array, the example around
        eps: the radius of the ball
        p: the type of Lp norm (p=2, 1, np.inf)
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        numpy array, vector with lower bounds of the range of values
        taken by the model inside the ball
    """

    return get_range_noise(model=model, x=x, eps=eps, p=p, batch_size=batch_size)[1]


# get upper bound of a sample with bounded noise
def get_range_noise(
    model: Union[keras.Model, DecomonModel],
    x: npt.NDArray[np.float_],
    eps: float,
    p: float = np.inf,
    batch_size: int = -1,
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Bounds the output of a model in an Lp Ball

    Args:
        model: either a Keras model or a Decomon model
        x: numpy array, the example around
        eps: the radius of the ball
        p: the type of Lp norm (p=2, 1, np.inf)
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        2 numpy arrays, vector with upper andlower bounds
    of the range of values taken by the model inside the ball
    """

    # check that the model is a DecomonModel, else do the conversion
    perturbation_domain = BallDomain(p=p, eps=max(0, eps))

    if not isinstance(model, DecomonModel):
        decomon_model = clone(
            model,
            method=ConvertMethod.CROWN_FORWARD_HYBRID,
            perturbation_domain=perturbation_domain,
        )
    else:
        decomon_model = model
        if eps >= 0:
            decomon_model.set_domain(perturbation_domain)

    # reshape x_mmin, x_max
    input_shape = list(decomon_model.input_shape[1:])
    input_dim = np.prod(input_shape)
    x_reshaped = x + 0 * x
    x_reshaped = x_reshaped.reshape([-1] + input_shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_reshaped) % batch_size > 0:
            r += 1
        x_list = [x_reshaped[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        results = [get_range_noise(decomon_model, x_list[i], eps=eps, p=p, batch_size=-1) for i in range(len(x_list))]

        return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])

    ibp = decomon_model.ibp
    affine = decomon_model.affine

    output = decomon_model.predict(x_reshaped, verbose=0)
    shape = list(output[-1].shape[1:])
    output_dim = np.prod(shape)

    x_reshaped = x_reshaped.reshape((len(x_reshaped), -1))
    ord = _get_dual_ord(p)

    if affine:
        if ibp:
            _, u_i, w_u_f, b_u_f, l_i, w_l_f, b_l_f = output[:7]
            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (-1, output_dim))
                l_i = np.reshape(l_i, (-1, output_dim))
        else:
            _, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (-1, input_dim, output_dim))
            b_u_f = np.reshape(b_u_f, (-1, output_dim))
            w_l_f = np.reshape(w_l_f, (-1, input_dim, output_dim))
            b_l_f = np.reshape(b_l_f, (-1, output_dim))

        u_f = eps * np.linalg.norm(w_u_f, ord=ord, axis=1) + np.sum(w_u_f * x_reshaped[:, :, None], 1) + b_u_f
        l_f = -eps * np.linalg.norm(w_l_f, ord=ord, axis=1) + np.sum(w_l_f * x_reshaped[:, :, None], 1) + b_l_f

    else:
        u_i = output[0]
        l_i = output[1]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (-1, output_dim))
            l_i = np.reshape(l_i, (-1, output_dim))
            ######

    if ibp and affine:
        u_out = np.minimum(u_i, u_f)
        l_out = np.maximum(l_i, l_f)
    elif ibp and not affine:
        u_out = u_i
        l_out = l_i
    elif not ibp and affine:
        u_out = u_f
        l_out = l_f
    else:
        raise NotImplementedError("not ibp and not affine not implemented")

    if len(shape) > 1:
        u_out = np.reshape(u_out, [-1] + shape)
        l_out = np.reshape(l_out, [-1] + shape)

    return u_out, l_out


def refine_boxes(
    x_min: npt.NDArray[np.float_], x_max: npt.NDArray[np.float_], n_sub_boxes: int = 10
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    # flatten x_min and x_max
    shape = list(x_min.shape[1:])
    output_dim = np.prod(shape)
    if len(shape) > 1:
        x_min = np.reshape(x_min, (-1, output_dim))
        x_max = np.reshape(x_min, (-1, output_dim))
    # x_min (None, n)
    # x_max (None, n)
    n = x_min.shape[-1]
    X_min = np.zeros((len(x_min), 1, n)) + x_min[:, None]
    X_max = np.zeros((len(x_max), 1, n)) + x_max[:, None]

    def split(
        x_min: npt.NDArray[np.float_], x_max: npt.NDArray[np.float_], j: npt.NDArray[np.int_]
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        n_0 = len(x_min)
        n_k = x_min.shape[1]

        x_min_reshaped = np.zeros((n_0, 2, n_k, n)) + x_min[:, None]
        x_max_reshaped = np.zeros((n_0, 2, n_k, n)) + x_max[:, None]

        index_0 = np.arange(n_0)
        index_k = np.arange(n_k)
        mid_x = (x_min + x_max) / 2.0
        split_value = np.array([mid_x[i, index_k, j[i]] for i in index_0])

        for i in index_0:
            x_min_reshaped[i, 1, index_k, j[i]] = split_value[i]
            x_max_reshaped[i, 0, index_k, j[i]] = split_value[i]

        return x_min_reshaped.reshape((-1, 2 * n_k, n)), x_max_reshaped.reshape((-1, 2 * n_k, n))

    # init
    n_sub = X_min.shape[1]
    while 2 * n_sub <= n_sub_boxes:
        j = np.argmax(X_max - X_min, -1)
        X_min, X_max = split(X_min, X_max, j)
        n_sub = X_min.shape[1]

    if len(shape) > 1:
        X_min = np.reshape(X_min, [-1, n_sub] + shape)
        X_max = np.reshape(X_max, [-1, n_sub] + shape)

    return X_min, X_max


def refine_box(
    func: Callable[..., npt.NDArray[np.float_]],
    model: Union[keras.Model, DecomonModel],
    x_min: npt.NDArray[np.float_],
    x_max: npt.NDArray[np.float_],
    n_split: int,
    source_labels: Optional[npt.NDArray[np.int_]] = None,
    target_labels: Optional[npt.NDArray[np.int_]] = None,
    batch_size: int = -1,
    random: bool = True,
) -> npt.NDArray[np.float_]:
    if func.__name__ not in [
        elem.__name__ for elem in [get_upper_box, get_lower_box, get_adv_box, check_adv_box, get_range_box]
    ]:
        raise NotImplementedError()

    # check that the model is a DecomonModel, else do the conversion
    if not isinstance(model, DecomonModel):
        decomon_model = clone(model)
    else:
        assert isinstance(model.perturbation_domain, (BoxDomain, GridDomain))
        decomon_model = model

    # reshape x_mmin, x_max
    input_shape = list(decomon_model.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_min = x_min.reshape((-1, input_dim))
    x_max = x_max.reshape((-1, input_dim))

    n_x = len(x_min)

    # split
    x_min_split = np.zeros((n_x, n_split, input_dim))
    x_max_split = np.zeros((n_x, n_split, input_dim))

    x_min_split[:, 0] = x_min
    x_max_split[:, 0] = x_max

    index_max = np.zeros((n_x, n_split, input_dim))
    # init
    index_max[:, 0] = x_max - x_min

    maximize = True
    if func.__name__ == get_lower_box.__name__:
        maximize = False

    def priv_func(x_min_split: npt.NDArray[np.float_], x_max_split: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        if func.__name__ in [elem.__name__ for elem in [get_upper_box, get_lower_box, get_range_box]]:
            results = func(decomon_model, x_min=x_min_split, x_max=x_max_split, batch_size=batch_size)

            if func.__name__ == get_upper_box.__name__:
                return results.reshape((n_x, n_split, -1))
            if func.__name__ == get_lower_box.__name__:
                return results.reshape((n_x, n_split, -1))

        if func.__name__ in [elem.__name__ for elem in [get_adv_box, check_adv_box]]:
            results = func(
                decomon_model,
                x_min=x_min_split,
                x_max=x_max_split,
                source_labels=source_labels,
                target_labels=target_labels,
                batch_size=batch_size,
            )

            return results.reshape((n_x, n_split))

        raise NotImplementedError()

    for n_i in range(n_x):
        count = 1

        while count < n_split:
            if not random:
                i = int(np.argmax(np.max(index_max[n_i, :count], -1)))
                j = int(np.argmax(index_max[n_i, i]))
            else:
                i = np.random.randint(count - 1)
                j = np.random.randint(input_dim)

            z_min = x_min_split[n_i, i] + 0.0
            z_max = x_max_split[n_i, i] + 0.0
            x_min_split[n_i, count] = z_min + 0.0
            x_max_split[n_i, count] = z_max + 0.0

            x_max_split[n_i, i, j] = (z_min[j] + z_max[j]) / 2.0
            x_min_split[n_i, count, j] = (z_min[j] + z_max[j]) / 2.0

            index_max[n_i, count] = index_max[n_i, i]
            index_max[n_i, i, j] /= 2.0
            index_max[n_i, count, j] /= 2.0

            count += 1

    x_min_split = x_min_split.reshape((-1, input_dim))
    x_max_split = x_max_split.reshape((-1, input_dim))

    results = priv_func(x_min_split, x_max_split)
    if maximize:
        return np.max(results, 1)
    else:
        return np.min(results, 1)


### adversarial robustness Lp norm
def get_adv_noise(
    model: Union[keras.Model, DecomonModel],
    x: npt.NDArray[np.float_],
    source_labels: LabelType,
    eps: float = 0.0,
    p: float = np.inf,
    target_labels: Optional[LabelType] = None,
    batch_size: int = -1,
) -> npt.NDArray[np.float_]:
    """if the output is negative, then it is a formal guarantee that there is no adversarial examples

    Args:
        model: either a Keras model or a Decomon model
        x_min: numpy array for the extremal lower corner of the boxes
        x_max: numpy array for the extremal upper corner of the boxes
        source_labels: the list of label that should be predicted all
            the time in the box (either an integer, either an array that
            can contain multiple source labels for each sample)
        target_labels: the list of label that should never be predicted
            in the box (either an integer, either an array that can
            contain multiple target labels for each sample)
        batch_size: for computational efficiency, one can split the
            calls to minibatches

    Returns:
        numpy array, vector with upper bounds for adversarial attacks
    """

    perturbation_domain = BallDomain(p=p, eps=max(0, eps))

    # check that the model is a DecomonModel, else do the conversion
    if not isinstance(model, DecomonModel):
        decomon_model = clone(model, perturbation_domain=perturbation_domain)
    else:
        decomon_model = model
        if eps >= 0:
            decomon_model.set_domain(perturbation_domain)

    eps = model.perturbation_domain.eps

    # reshape x_mmin, x_max
    input_shape = list(decomon_model.input_shape[1:])
    x_reshaped = x + 0 * x
    x_reshaped = x_reshaped.reshape([-1] + input_shape)
    n_split = 1
    n_batch = len(x_reshaped)

    input_shape = list(decomon_model.input_shape[1:])
    x_reshaped = x + 0 * x
    x_reshaped = x_reshaped.reshape([-1] + input_shape)

    if isinstance(source_labels, (int, np.int_)):
        source_labels = np.zeros((n_batch, 1), dtype=np.int_) + source_labels
    source_labels = np.array(source_labels).reshape((n_batch, -1)).astype(np.int_)

    if target_labels is not None:
        if isinstance(target_labels, (int, np.int_)):
            target_labels = np.zeros((n_batch, 1), dtype=np.int_) + source_labels
        target_labels = np.array(target_labels).reshape((n_batch, -1)).astype(np.int_)

    if batch_size > 0:
        # split
        r = 0
        if len(x_reshaped) % batch_size > 0:
            r += 1
        x_list = [x_reshaped[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)]
        source_labels_list = [
            source_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)
        ]
        target_labels_list: List[Optional[npt.NDArray[np.int_]]]
        if (
            (target_labels is not None)
            and (not isinstance(target_labels, int))
            and (str(target_labels.dtype)[:3] != "int")
        ):
            target_labels_list = [
                target_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_reshaped) // batch_size + r)
            ]
        else:
            target_labels_list = [target_labels] * (len(x_reshaped) // batch_size + r)

        results = [
            get_adv_noise(
                decomon_model,
                x_list[i],
                source_labels=source_labels_list[i],
                eps=eps,
                p=p,
                target_labels=target_labels_list[i],
                batch_size=-1,
            )
            for i in range(len(x_list))
        ]

        return np.concatenate(results)
    else:
        ibp = decomon_model.ibp
        affine = decomon_model.affine
        output = decomon_model.predict(x_reshaped, verbose=0)

        def get_ibp_score(
            u_c: npt.NDArray[np.float_],
            l_c: npt.NDArray[np.float_],
            source_tensor: npt.NDArray[np.int_],
            target_tensor: Optional[npt.NDArray[np.int_]] = None,
        ) -> npt.NDArray[np.float_]:
            if target_tensor is None:
                target_tensor = 1 - source_tensor

            shape = np.prod(u_c.shape[1:])

            t_tensor_reshaped = np.reshape(target_tensor, (-1, shape))
            s_tensor_reshaped = np.reshape(source_tensor, (-1, shape))

            # add penalties on biases
            upper = u_c[:, :, None] - l_c[:, None]
            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_reshaped[:, :, None] * s_tensor_reshaped[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        def get_affine_score(
            z_tensor: npt.NDArray[np.float_],
            w_u: npt.NDArray[np.float_],
            b_u: npt.NDArray[np.float_],
            w_l: npt.NDArray[np.float_],
            b_l: npt.NDArray[np.float_],
            source_tensor: npt.NDArray[np.int_],
            target_tensor: Optional[npt.NDArray[np.int_]] = None,
        ) -> npt.NDArray[np.float_]:
            if target_tensor is None:
                target_tensor = 1 - source_tensor

            n_dim = w_u.shape[1]
            shape = np.prod(b_u.shape[1:])
            w_u_reshaped = np.reshape(w_u, (-1, n_dim, shape, 1))
            w_l_reshaped = np.reshape(w_l, (-1, n_dim, 1, shape))
            b_u_reshaped = np.reshape(b_u, (-1, shape, 1))
            b_l_reshaped = np.reshape(b_l, (-1, 1, shape))

            t_tensor_reshaped = np.reshape(target_tensor, (-1, shape))
            s_tensor_reshaped = np.reshape(source_tensor, (-1, shape))

            w_u_f = w_u_reshaped - w_l_reshaped
            b_u_f = b_u_reshaped - b_l_reshaped

            # add penalties on biases

            # compute upper with lp norm

            if len(z_tensor.shape) > 2:
                z_tensor = np.reshape(z_tensor, (len(z_tensor), -1))

            upper_0 = np.sum(w_u_f * z_tensor[:, :, None, None], 1) + b_u_f
            # compute dual norm
            if p == 2:
                upper_1 = eps * np.sum(w_u_f**2, 1)
            elif p == 1:
                upper_1 = eps * np.max(np.abs(w_u_f), 1)
            elif p == np.inf:
                upper_1 = eps * np.sum(np.abs(w_u_f), 1)
            else:
                raise ValueError(f"p must be equal to 1, 2, or np.inf, unknown value {p}.")

            upper = upper_0 + upper_1

            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_reshaped[:, :, None] * s_tensor_reshaped[:, None, :]
            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        if ibp and affine:
            z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:7]
        if not ibp and affine:
            z, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]
        if ibp and not affine:
            u_c, l_c = output[:2]
        else:
            raise NotImplementedError("not ibp and not affine not implemented")

        if ibp:
            adv_ibp = get_ibp_score(u_c, l_c, source_labels, target_labels)
        if affine:
            adv_f = get_affine_score(z, w_u_f, b_u_f, w_l_f, b_l_f, source_labels, target_labels)

        if ibp and not affine:
            adv_score = adv_ibp
        elif ibp and affine:
            adv_score = np.minimum(adv_ibp, adv_f)
        elif not ibp and affine:
            adv_score = adv_f
        else:
            raise NotImplementedError("not ibp and not affine not implemented")

        if n_split > 1:
            adv_score = np.max(np.reshape(adv_score, (-1, n_split)), -1)

        return adv_score
