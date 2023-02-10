import numpy as np

from decomon.layers.core import Ball, Box, Grid
from decomon.models.convert import clone as convert
from decomon.models.models import DecomonModel


def _get_dual_ord(p):
    if p == np.inf:
        return 1
    elif p == 1:
        return np.inf
    elif p == 2:
        return 2
    else:
        raise ValueError(f"p must be equal to 1, 2, or np.inf, unknown value {p}.")


##### ADVERSARIAL ROBUSTTNESS #####
def get_adv_box(
    model,
    x_min,
    x_max,
    source_labels,
    target_labels=None,
    batch_size=-1,
    n_sub_boxes=1,
):
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
        model_ = convert(model, ibp=True, forward=True)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] in [Box.name, Grid.name]
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
    if model_.backward_bounds:
        input_shape = list(model_.input_shape[0][2:])
    else:
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
            [get_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1) for i in range(len(X_min_))]
        )

    else:

        IBP = model_.IBP
        forward = model_.forward
        n_label = source_labels.shape[-1]

        # two possitible cases: the model improves the bound based on the knowledge of the labels
        if model_.backward_bounds:
            C = np.diag([1] * n_label)[None] - source_labels[:, :, None]
            output = model_.predict([z, C])
        else:
            output = model_.predict(z)

        def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None, backward=False):

            if target_tensor is None:
                target_tensor = 1 - source_tensor

            shape = np.prod(u_c.shape[1:])

            t_tensor_ = np.reshape(target_tensor, (-1, shape))
            s_tensor_ = np.reshape(source_tensor, (-1, shape))

            # add penalties on biases
            if backward:
                return np.max(u_c, -1)
            else:
                upper = u_c[:, :, None] - l_c[:, None]
            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_[:, :, None] * s_tensor_[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        def get_forward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None, backward=False):

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

            if not backward:
                w_u_f = w_u_ - w_l_
                b_u_f = b_u_ - b_l_
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
            discard_mask_s = t_tensor_[:, :, None] * s_tensor_[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

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
            adv_ibp = get_ibp_score(u_c, l_c, source_labels, target_labels, backward=model_.backward_bounds)
        if forward:
            adv_f = get_forward_score(
                z, w_u_f, b_u_f, w_l_f, b_l_f, source_labels, target_labels, backward=model_.backward_bounds
            )

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


def check_adv_box(model, x_min, x_max, source_labels, target_labels=None, batch_size=-1):
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
        model_ = convert(model, ibp=True, forward=True)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] in [Box.name, Grid.name]
        model_ = model

    IBP = model_.IBP
    forward = model_.forward

    n_split = 1
    n_batch = len(x_min)
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

        return np.concatenate(
            [check_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1) for i in range(len(X_min_))]
        )

    else:

        output = model_.predict(z)

        if not forward:
            # translate  into forward information
            u_c = output[0]
            w_u = 0 * u_c[:, None] + np.zeros((1, input_dim, 1))
            output = [z, w_u, u_c, w_u, output[-1]]
            IBP = False
            forward = True

        def get_forward_sample(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

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
            upper = (
                np.sum(np.maximum(w_u_f, 0) * z_tensor[:, 1, :, None, None], 1)
                + np.sum(np.minimum(w_u_f, 0) * z_tensor[:, 0, :, None, None], 1)
                + b_u_f
            )  # (-1, shape, shape)
            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_[:, :, None] * s_tensor_[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        if IBP:
            z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:7]
        else:
            z, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]

        return get_forward_sample(z, w_l_f, b_l_f, w_u_f, b_u_f, source_labels)


#### FORMAL BOUNDS ######
def get_upper_box(model, x_min, x_max, batch_size=-1, n_sub_boxes=1):
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

    if np.min(x_max - x_min) < 0:
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    if not (isinstance(model, DecomonModel)):
        model_ = convert(model, ibp=True, forward=True)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] in [Box.name, Grid.name]
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

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_min_ = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        X_max_ = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]

        results = [get_upper_box(model_, X_min_[i], X_max_[i], -1) for i in range(len(X_min_))]

        u_ = np.concatenate(results)
    else:
        IBP = model_.IBP
        forward = model_.forward
        output = model_.predict(z)
        shape = output[-1].shape[1:]
        shape_ = np.prod(shape)

        if forward:
            if IBP:
                _, u_i, w_u_f, b_u_f, _, _, _ = output[:7]
                if len(u_i.shape) > 2:
                    u_i = np.reshape(u_i, (-1, shape_))

            else:
                _, w_u_f, b_u_f, _, _ = output[:5]

            # reshape if necessary
            if len(w_u_f.shape) > 3:
                w_u_f = np.reshape(w_u_f, (-1, input_dim, shape_))
                b_u_f = np.reshape(b_u_f, (-1, input_dim, shape_))

            u_f = (
                np.sum(np.maximum(w_u_f, 0) * x_max[:, 0, :, None], 1)
                + np.sum(np.minimum(w_u_f, 0) * x_min[:, 0, :, None], 1)
                + b_u_f
            )

        else:
            u_i = output[0]
            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (-1, shape_))
                ######

        if IBP and forward:
            u_ = np.minimum(u_i, u_f)
        elif IBP and not forward:
            u_ = u_i
        elif not IBP and forward:
            u_ = u_f
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if len(shape) > 1:
            u_ = np.reshape(u_, [-1] + list(shape))

    if n_split > 1:
        u_ = np.max(np.reshape(u_, (-1, n_split)), -1)

    return u_


def get_lower_box(model, x_min, x_max, batch_size=-1, n_sub_boxes=1):
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

    if np.min(x_max - x_min) < 0:
        raise UserWarning("Inconsistency Error: x_max < x_min")

    # check that the model is a DecomonModel, else do the conversion
    if not (isinstance(model, DecomonModel)):
        model_ = convert(model, ibp=True, forward=True)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] in [Box.name, Grid.name]
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

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_min_ = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        X_max_ = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]

        results = [get_lower_box(model_, X_min_[i], X_max_[i], -1) for i in range(len(X_min_))]

        l_ = np.concatenate(results)
    else:
        IBP = model_.IBP
        forward = model_.forward

        output = model_.predict(z)
        shape = list(output[-1].shape[1:])
        shape_ = np.prod(shape)

        if forward:
            if IBP:
                _, _, _, _, l_i, w_l_f, b_l_f = output[:7]
                if len(l_i.shape) > 2:
                    l_i = np.reshape(l_i, (-1, shape_))
            else:
                _, _, _, w_l_f, b_l_f = output[:5]

            # reshape if necessary
            if len(w_l_f.shape) > 3:
                w_l_f = np.reshape(w_l_f, (-1, input_dim, shape_))
                b_l_f = np.reshape(b_l_f, (-1, input_dim, shape_))

            l_f = (
                np.sum(np.maximum(w_l_f, 0) * x_min[:, 0, :, None], 1)
                + np.sum(np.minimum(w_l_f, 0) * x_max[:, 0, :, None], 1)
                + b_l_f
            )

        else:
            l_i = output[1]
            if len(l_i.shape) > 2:
                l_i = np.reshape(l_i, (-1, shape_))
                ######

        if IBP and forward:
            l_ = np.maximum(l_i, l_f)
        elif IBP and not forward:
            l_ = l_i
        elif not IBP and forward:
            l_ = l_f
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if len(shape) > 1:
            l_ = np.reshape(l_, [-1] + list(shape_))

    if n_split > 1:
        l_ = np.min(np.reshape(l_, (-1, n_split)), -1)
    return l_


def get_range_box(model, x_min, x_max, batch_size=-1, n_sub_boxes=1):
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
        model_ = convert(model, ibp=True, forward=True)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] in [Box.name, Grid.name]
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

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_min_ = [x_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        X_max_ = [x_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        results = [get_range_box(model_, X_min_[i], X_max_[i], -1) for i in range(len(X_min_))]

        u_ = [r[0] for r in results]
        l_ = [r[1] for r in results]

        u_ = np.concatenate(u_)
        l_ = np.concatenate(l_)
    else:
        IBP = model_.IBP
        forward = model_.forward

        output = model_.predict(z)
        shape = list(output[-1].shape[1:])
        shape_ = np.prod(shape)

        if forward:
            if IBP:
                _, u_i, w_u_f, b_u_f, l_i, w_l_f, b_l_f = output[:7]
                if len(u_i.shape) > 2:
                    u_i = np.reshape(u_i, (-1, u_i.shape_))
                    l_i = np.reshape(l_i, (-1, l_i.shape_))
            else:
                _, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]

            # reshape if necessary
            if len(w_u_f.shape) > 3:
                w_u_f = np.reshape(w_u_f, (-1, input_dim, shape_))
                w_l_f = np.reshape(w_l_f, (-1, input_dim, shape_))
                b_u_f = np.reshape(b_u_f, (-1, shape_))
                b_l_f = np.reshape(b_l_f, (-1, shape_))

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
                u_i = np.reshape(u_i, (-1, u_i.shape_))
                l_i = np.reshape(l_i, (-1, l_i.shape_))

        if IBP and forward:
            u_ = np.minimum(u_i, u_f)
            l_ = np.maximum(l_i, l_f)
        elif IBP and not forward:
            u_ = u_i
            l_ = l_i
        elif not IBP and forward:
            u_ = u_f
            l_ = l_f
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        #####
        if len(shape) > 1:
            u_ = np.reshape(u_, [-1] + shape)
            l_ = np.reshape(l_, [-1] + shape)

    if n_split > 1:
        u_ = np.max(np.reshape(u_, (-1, n_split)), -1)
        l_ = np.min(np.reshape(l_, (-1, n_split)), -1)
    return u_, l_


# get upper bound of a sample with bounded noise
def get_upper_noise(model, x, eps=0, p=np.inf, batch_size=-1):
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
    # check that the model is a DecomonModel, else do the conversion
    convex_domain = {"name": Ball.name, "p": p, "eps": max(0, eps)}

    # check that the model is a DecomonModel, else do the conversion
    if not isinstance(model, DecomonModel):
        model_ = convert(model, method="crown-hybrid", convex_domain=convex_domain, ibp=True, forward=False)
    else:
        model_ = model
        if eps >= 0:
            model_.set_domain(convex_domain)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[1:])
    input_dim = np.prod(input_shape)
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_ = [x_[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        results = [get_upper_noise(model_, X_[i], eps=eps, p=p, batch_size=-1) for i in range(len(X_))]

        return np.concatenate(results)

    IBP = model_.IBP
    forward = model_.forward
    output = model_.predict(x_)
    shape = output[-1].shape[1:]
    shape_ = np.prod(shape)

    x_ = x_.reshape((len(x_), -1))
    ord = _get_dual_ord(p)

    if forward:
        if IBP:
            _, u_i, w_u_f, b_u_f, _, _, _ = output[:7]
            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (-1, input_dim))
        else:
            _, w_u_f, b_u_f, _, _ = output[:5]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (-1, input_dim, shape_))
            b_u_f = np.reshape(b_u_f, (-1, shape_))

        u_f = eps * np.linalg.norm(w_u_f, ord=ord, axis=1) + np.sum(w_u_f * x_[:, :, None], 1) + b_u_f

    else:
        u_i = output[0]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (-1, input_dim))
            ######

    if IBP and forward:
        u_ = np.minimum(u_i, u_f)
    elif IBP and not forward:
        u_ = u_i
    elif not IBP and forward:
        u_ = u_f
    else:
        raise NotImplementedError("not IBP and not forward not implemented")

    if len(shape) > 1:
        u_ = np.reshape(u_ + [-1] + list(shape))

    return u_


# get upper bound of a sample with bounded noise
def get_lower_noise(model, x, eps, p=np.inf, batch_size=-1):
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
    # check that the model is a DecomonModel, else do the conversion
    convex_domain = {"name": Ball.name, "p": p, "eps": max(0, eps)}

    if not isinstance(model, DecomonModel):
        model_ = convert(model, method="crown-hybrid", convex_domain=convex_domain, ibp=True, forward=False)
    else:
        model_ = model
        if eps >= 0:
            model_.set_domain(convex_domain)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[1:])
    input_dim = np.prod(input_shape)
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_ = [x_[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        results = [get_lower_noise(model_, X_[i], eps=eps, p=p, batch_size=-1) for i in range(len(X_))]

        return np.concatenate(results)

    IBP = model_.IBP
    forward = model_.forward
    output = model_.predict(x_)
    shape = list(output[-1].shape[1:])
    shape_ = np.prod(shape)

    x_ = x_.reshape((len(x_), -1))
    ord = _get_dual_ord(p)

    if forward:
        if IBP:
            _, _, _, _, l_i, w_l_f, b_l_f = output[:7]
            if len(l_i.shape) > 2:
                l_i = np.reshape(l_i, (-1, shape_))
        else:
            _, _, _, w_l_f, b_l_f = output[:5]

        # reshape if necessary
        if len(w_l_f.shape) > 3:
            w_l_f = np.reshape(w_l_f, (-1, input_dim, shape_))
            b_l_f = np.reshape(b_l_f, (-1, shape_))
        l_f = -eps * np.linalg.norm(w_l_f, ord=ord, axis=1) + np.sum(w_l_f * x_[:, :, None], 1) + b_l_f

    else:
        l_i = output[1]
        if len(l_i.shape) > 2:
            l_i = np.reshape(l_i, (-1, shape_))
            ######

    if IBP and forward:
        l_ = np.maximum(l_i, l_f)
    elif IBP and not forward:
        l_ = l_i
    elif not IBP and forward:
        l_ = l_f
    else:
        raise NotImplementedError("not IBP and not forward not implemented")

    if len(shape) > 1:
        l_ = np.reshape(l_, [-1] + shape)

    return l_


# get upper bound of a sample with bounded noise
def get_range_noise(model, x, eps, p=np.inf, batch_size=-1):
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
    convex_domain = {"name": Ball.name, "p": p, "eps": max(0, eps)}

    if not isinstance(model, DecomonModel):
        model_ = convert(model, method="crown-hybrid", convex_domain=convex_domain, ibp=True, forward=False)
    else:
        model_ = model
        if eps >= 0:
            model_.set_domain(convex_domain)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[1:])
    input_dim = np.prod(input_shape)
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_ = [x_[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        results = [get_range_noise(model_, X_[i], eps=eps, p=p, batch_size=-1) for i in range(len(X_))]

        u_ = [r[0] for r in results]
        l_ = [r[1] for r in results]

        return np.concatenate(u_), np.concatenate(l_)

    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict(x_)
    shape = list(output[-1].shape[1:])
    shape_ = np.prod(shape)

    x_ = x_.reshape((len(x_), -1))
    ord = _get_dual_ord(p)

    if forward:
        if IBP:
            _, u_i, w_u_f, b_u_f, l_i, w_l_f, b_l_f = output[:7]
            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (-1, shape_))
                l_i = np.reshape(l_i, (-1, shape_))
        else:
            _, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (-1, input_dim, shape_))
            b_u_f = np.reshape(b_u_f, (-1, shape_))
            w_l_f = np.reshape(w_l_f, (-1, input_dim, shape_))
            b_l_f = np.reshape(b_l_f, (-1, shape_))

        u_f = eps * np.linalg.norm(w_u_f, ord=ord, axis=1) + np.sum(w_u_f * x_[:, :, None], 1) + b_u_f
        l_f = -eps * np.linalg.norm(w_l_f, ord=ord, axis=1) + np.sum(w_l_f * x_[:, :, None], 1) + b_l_f

    else:
        u_i = output[0]
        l_i = output[1]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (-1, shape_))
            l_i = np.reshape(l_i, (-1, shape_))
            ######

    if IBP and forward:
        u_ = np.minimum(u_i, u_f)
        l_ = np.maximum(l_i, l_f)
    elif IBP and not forward:
        u_ = u_i
        l_ = l_i
    elif not IBP and forward:
        u_ = u_f
        l_ = l_f
    else:
        raise NotImplementedError("not IBP and not forward not implemented")

    if len(shape) > 1:
        u_ = np.reshape(u_, [-1] + shape)
        l_ = np.reshape(l_, [-1] + shape)

    return u_, l_


def refine_boxes(x_min, x_max, n_sub_boxes=10):

    # flatten x_min and x_max
    shape = list(x_min.shape[1:])
    shape_ = np.prod(shape)
    if len(shape) > 1:
        x_min = np.reshape(x_min, (-1, shape_))
        x_max = np.reshape(x_min, (-1, shape_))
    # x_min (None, n)
    # x_max (None, n)
    n = x_min.shape[-1]
    X_min = np.zeros((len(x_min), 1, n)) + x_min[:, None]
    X_max = np.zeros((len(x_max), 1, n)) + x_max[:, None]

    def split(x_min_, x_max_, j):
        n_0 = len(x_min_)
        n_k = x_min_.shape[1]

        X_min_ = np.zeros((n_0, 2, n_k, n)) + x_min_[:, None]
        X_max_ = np.zeros((n_0, 2, n_k, n)) + x_max_[:, None]

        index_0 = np.arange(n_0)
        index_k = np.arange(n_k)
        mid_x = (x_min_ + x_max_) / 2.0
        split_value = np.array([mid_x[i, index_k, j[i]] for i in index_0])

        for i in index_0:
            X_min_[i, 1, index_k, j[i]] = split_value[i]
            X_max_[i, 0, index_k, j[i]] = split_value[i]

        return X_min_.reshape((-1, 2 * n_k, n)), X_max_.reshape((-1, 2 * n_k, n))

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


def refine_box(func, model, x_min, x_max, n_split, source_labels=None, target_labels=None, batch_size=-1, random=True):

    if func.__name__ not in [
        elem.__name__ for elem in [get_upper_box, get_lower_box, get_adv_box, check_adv_box, get_range_box]
    ]:
        raise NotImplementedError()

    # check that the model is a DecomonModel, else do the conversion
    if not isinstance(model, DecomonModel):
        model_ = convert(model)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] in [Box.name, Grid.name]
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[2:])
    input_dim = np.prod(input_shape)
    x_min = x_min.reshape((-1, input_dim))
    x_max = x_max.reshape((-1, input_dim))

    n_x = len(x_min)

    # split
    X_min = np.zeros((n_x, n_split, input_dim))
    X_max = np.zeros((n_x, n_split, input_dim))

    X_min[:, 0] = x_min
    X_max[:, 0] = x_max

    index_max = np.zeros((n_x, n_split, input_dim))
    # init
    index_max[:, 0] = x_max - x_min

    maximize = True
    if func.__name__ == get_lower_box.__name__:
        maximize = False

    def priv_func(X_min, X_max):
        if func.__name__ in [elem.__name__ for elem in [get_upper_box, get_lower_box, get_range_box]]:

            results = func(model_, x_min=X_min, x_max=X_max, batch_size=batch_size)

            if func.__name__ == get_upper_box.__name__:
                return results.reshape((n_x, n_split, -1))
            if func.__name__ == get_lower_box.__name__:
                return results.reshape((n_x, n_split, -1))

        if func.__name__ in [elem.__name__ for elem in [get_adv_box, check_adv_box]]:

            results = func(
                model_,
                x_min=X_min,
                x_max=X_max,
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
                i = np.argmax(np.max(index_max[n_i, :count], -1))
                j = np.argmax(index_max[n_i, i])
            else:
                i = np.random.randint(count - 1)
                j = np.random.randint(input_dim)

            z_min = X_min[n_i, i] + 0.0
            z_max = X_max[n_i, i] + 0.0
            X_min[n_i, count] = z_min + 0.0
            X_max[n_i, count] = z_max + 0.0

            X_max[n_i, i, j] = (z_min[j] + z_max[j]) / 2.0
            X_min[n_i, count, j] = (z_min[j] + z_max[j]) / 2.0

            index_max[n_i, count] = index_max[n_i, i]
            index_max[n_i, i, j] /= 2.0
            index_max[n_i, count, j] /= 2.0

            count += 1
            X_min_ = X_min.reshape((-1, input_dim))
            X_max_ = X_max.reshape((-1, input_dim))

    X_min_ = X_min.reshape((-1, input_dim))
    X_max_ = X_max.reshape((-1, input_dim))

    results = priv_func(X_min_, X_max_)
    if maximize:
        return np.max(results, 1)
    else:
        return np.min(results, 1)


### adversarial robustness Lp norm
def get_adv_noise(
    model,
    x,
    source_labels,
    eps=0,
    p=np.inf,
    target_labels=None,
    batch_size=-1,
):
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

    convex_domain = {"name": Ball.name, "p": p, "eps": max(0, eps)}

    # check that the model is a DecomonModel, else do the conversion
    if not isinstance(model, DecomonModel):
        model_ = convert(model, convex_domain=convex_domain)
    else:
        model_ = model
        if eps >= 0:
            model_.set_domain(convex_domain)

    eps = model.convex_domain["eps"]

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[1:])
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)
    n_split = 1
    n_batch = len(x_)

    input_shape = list(model_.input_shape[1:])
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)

    if isinstance(source_labels, (int, np.int64)):
        source_labels = np.zeros((n_batch, 1)) + source_labels

    if isinstance(source_labels, list):
        source_labels = np.array(source_labels).reshape((n_batch, -1))

    source_labels = source_labels.reshape((n_batch, -1))
    source_labels = source_labels.astype("int64")

    if target_labels is not None:
        target_labels = target_labels.reshape((n_batch, -1))
        target_labels = target_labels.astype("int64")

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_ = [x_[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        S_ = [source_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        if (
            (target_labels is not None)
            and (not isinstance(target_labels, int))
            and (str(target_labels.dtype)[:3] != "int")
        ):
            T_ = [target_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        else:
            T_ = [target_labels] * (len(x_) // batch_size + r)

        results = [
            get_adv_noise(model_, X_[i], source_labels=S_[i], eps=eps, p=p, target_labels=T_[i], batch_size=-1)
            for i in range(len(X_))
        ]

        return np.concatenate(results)
    else:

        IBP = model_.IBP
        forward = model_.forward
        output = model_.predict(x_)

        def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

            if target_tensor is None:
                target_tensor = 1 - source_tensor

            shape = np.prod(u_c.shape[1:])

            t_tensor_ = np.reshape(target_tensor, (-1, shape))
            s_tensor_ = np.reshape(source_tensor, (-1, shape))

            # add penalties on biases
            upper = u_c[:, :, None] - l_c[:, None]
            const = upper.max() - upper.min()
            discard_mask_s = t_tensor_[:, :, None] * s_tensor_[:, None, :]

            upper -= (1 - discard_mask_s) * (const + 0.1)

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
            discard_mask_s = t_tensor_[:, :, None] * s_tensor_[:, None, :]
            upper -= (1 - discard_mask_s) * (const + 0.1)

            return np.max(np.max(upper, -2), -1)

        if IBP and forward:
            z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:7]
        if not IBP and forward:
            z, w_u_f, b_u_f, w_l_f, b_l_f = output[:5]
        if IBP and not forward:
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
