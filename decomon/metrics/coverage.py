## Measuring the percentage of the input space that could be discarded as we are able to prove the property on it
import numpy as np

from decomon.layers.core import Ball, Box
from decomon.models.convert import clone as convert
from decomon.models.decomon_sequential import Backward, Forward
from decomon.models.models import DecomonModel
from decomon.wrapper import get_adv_box, refine_boxes


def get_adv_coverage_box(
    model,
    x_min,
    x_max,
    source_labels,
    target_labels=None,
    batch_size=-1,
    n_sub_boxes=1,
    fast=True,
):
    """
    if the output is negative, then it is a formal guarantee that there is no adversarial examples
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param source_labels: the list of label that should be predicted all the time in the box (either an integer, either an array that can contain multiple source labels for each sample)
    :param target_labels: the list of label that should never be predicted in the box (either an integer, either an array that can contain multiple target labels for each sample)
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: numpy array, vector with upper bounds for adversarial attacks
    """
    if np.min(x_max - x_min) < 0:
        import pdb

        pdb.set_trace()
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

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
    input_shape = list(model_.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_ = x_min + 0 * x_min
    x_ = x_.reshape([-1] + input_shape)

    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)

    if isinstance(source_labels, (int, np.int64)):
        source_labels = np.zeros((n_batch, 1)) + source_labels

    if isinstance(source_labels, list):
        source_labels = np.array(source_labels).reshape((n_batch, -1))

    source_labels = source_labels.reshape((n_batch, -1))
    source_labels = source_labels.astype("int64")

    if target_labels is not None:
        target_labels = target_labels.reshape((n_batch, -1))
        target_labels = target_labels.astype("int64")

    if n_split > 1:
        shape = list(source_labels.shape[1:])
        source_labels = np.reshape(np.concatenate([source_labels[:, None]] * n_split, 1), [-1] + shape)
        if target_labels is not None:
            target_labels = np.reshape(np.concatenate([target_labels[:, None]] * n_split, 1), [-1] + shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_min_ = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        X_max_ = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        S_ = [source_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        if (
            (target_labels is not None)
            and (not isinstance(target_labels, int))
            and (str(target_labels.dtype)[:3] != "int")
        ):
            T_ = [target_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        else:
            T_ = [target_labels] * (len(x_) // batch_size + r)

        adv_score = np.concatenate(
            [get_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1, fast=fast) for i in range(len(X_min_))]
        )

    else:

        IBP = model_.IBP
        forward = model_.forward

        output = model_.predict(z)

        n_label = source_labels.shape[-1]

        def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1 - source_tensor

            shape = np.prod(u_c.shape[1:])

            t_tensor_ = np.reshape(target_tensor, (-1, shape))
            s_tensor_ = np.reshape(source_tensor, (-1, shape))

            # add penalties on biases
            upper = u_c[:, :, None] - l_c[:, None]
            const = upper.max() - upper.min()
            # upper = upper*s_tensor_[:,None, :] + (const+0.1)*(1. - s_tensor_[:,None,:])
            discard_mask_s = t_tensor_[:, :, None] * s_tensor_[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            # upper = upper * s_tensor_[:, :, None] - (const + 0.1) * (1. - s_tensor_[:, None,:])
            # upper = upper*t_tensor_[:,:,None ] - (const+0.1)*(1. - t_tensor_[:,None,:])

            """
                        if target_tensor is None:
                target_tensor = 1 - source_tensor

            shape = np.prod(u_c.shape[1:])
            u_c_ = np.reshape(u_c, (-1, shape))
            l_c_ = np.reshape(l_c, (-1, shape))

            t_tensor_ = np.reshape(target_tensor, (-1, shape))
            s_tensor_ = np.reshape(source_tensor, (-1, shape))

            #score_u = np.min(u_c_ * t_tensor_ + (u_c_.max() + 1e6) * (1 - t_tensor_), -1)
            score_u = np.max(u_c * t_tensor_ + (u_c_.min() - 0.1) * (1 - t_tensor_), -1)
            score_l = np.max(l_c * s_tensor_ + (l_c_.min() - 0.1) * (1 - s_tensor_), -1)

            return score_u - score_l
            """

            return np.max(np.max(upper, -2), -1)

        def get_forward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1 - source_tensor

            n_dim = w_u.shape[1]
            shape = np.prod(b_u.shape[1:])
            w_u_ = np.reshape(w_u, (-1, n_dim, shape, 1))
            w_l_ = np.reshape(w_l, (-1, n_dim, 1, shape))
            b_u_ = np.reshape(b_u, (-1, shape, 1))
            b_l_ = np.reshape(b_l, (-1, 1, shape))

            t_tensor_ = np.reshape(target_tensor, (-1, shape))
            s_tensor_ = np.reshape(source_tensor, (-1, shape))

            w_u_f = w_u_ - w_l_
            b_u_f = b_u_ - b_l_

            # add penalties on biases
            # invert x_max and x_min
            upper = (
                np.sum(np.maximum(w_u_f, 0) * z_tensor[:, 0, :, None, None], 1)
                + np.sum(np.minimum(w_u_f, 0) * z_tensor[:, 1, :, None, None], 1)
                + b_u_f
            )  # (-1, shape, shape)
            const = upper.max() - upper.min()
            # upper = upper*s_tensor_[:,None, :] + (const+0.1)*(1. - s_tensor_[:,None,:])
            discard_mask_s = t_tensor_[:, :, None] * s_tensor_[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            # upper = upper * s_tensor_[:, :, None] - (const + 0.1) * (1. - s_tensor_[:, None,:])
            # upper = upper*t_tensor_[:,:,None ] - (const+0.1)*(1. - t_tensor_[:,None,:])

            return np.max(np.max(upper, -2), -1)

        if IBP and forward:
            z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:7]
        elif not IBP and forward:
            z, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]
        elif IBP and not forward:
            u_c, l_c = output[:2]
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if IBP:
            adv_ibp = get_ibp_score(u_c, l_c, source_labels, target_labels)
        if forward:
            adv_f = get_forward_score(z, w_u_f, b_u_f, w_l_f, b_l_f, source_labels, target_labels)

        if IBP and not forward:
            adv_score = adv_ibp
        elif IBP and forward:
            adv_score = np.minimum(adv_ibp, adv_f)
        elif not IBP and forward:
            adv_score = adv_f
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

    if n_split > 1:
        adv_score = np.max(np.reshape(adv_score, (-1, n_split)), -1)

    return adv_score
