from .models import DecomonModel, convert
import numpy as np
from .layers.core import Ball, Box
from .models.decomon_sequential import Backward, Forward


##### ADVERSARIAL ROBUSTTNESS #####
def get_adv_box(model, x_min, x_max, source_labels, target_labels=None, batch_size=-1, fast=True):
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

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
    input_dim = np.prod(input_shape)
    x_ = x_min + 0 * x_min
    x_ = x_.reshape([-1] + input_shape)

    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)
    if isinstance(source_labels, int) or isinstance(source_labels, np.int64):
        source_labels = np.zeros((len(x_), 1)) + source_labels

    if isinstance(source_labels, list):
        source_labels = np.array(source_labels).reshape((len(x_), -1))

    source_labels = source_labels.reshape((len(x_), -1))
    source_labels = source_labels.astype("int64")

    if target_labels is not None:
        target_labels = target_labels.reshape((len(x_), -1))
        target_labels = target_labels.astype("int64")

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

        results = [get_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1, fast=fast) for i in range(len(X_min_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, z])

    n_label = source_labels.shape[-1]

    def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

        if target_tensor is None:
            target_tensor = 1 - source_tensor

        shape = np.prod(u_c.shape[1:])
        u_c_ = np.reshape(u_c, (-1, shape))
        l_c_ = np.reshape(l_c, (-1, shape))

        score_u = (
            u_c_ * target_tensor - np.expand_dims(np.min(l_c_ * source_tensor, -1), -1) - 1e6 * (1 - target_tensor)
        )

        return np.max(score_u, -1)

    def get_forward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

        if target_tensor is None:
            target_tensor = 1 - source_tensor

        n_dim = w_u.shape[1]
        shape = np.prod(b_u.shape[1:])
        w_u_ = np.reshape(w_u, (-1, n_dim, shape, 1))
        w_l_ = np.reshape(w_l, (-1, n_dim, 1, shape))
        b_u_ = np.reshape(b_u, (-1, shape, 1))
        b_l_ = np.reshape(b_l, (-1, 1, shape))

        w_u_f = w_u_ - w_l_
        b_u_f = b_u_ - b_l_

        # add penalties on biases
        b_u_f = b_u_f - 1e6 * (1 - source_tensor)[:, None, :]
        b_u_f = b_u_f - 1e6 * (1 - target_tensor)[:, :, None]

        upper = (
            np.sum(np.maximum(w_u_f, 0) * z_tensor[:, 1, :, None, None], 1)
            + np.sum(np.minimum(w_u_f, 0) * z_tensor[:, 0, :, None, None], 1)
            + b_u_f
        )

        return np.max(upper, (-1, -2))

    def get_backward_score(z_tensor, w_u, b_u, w_l, b_l, y_tensor, target_tensor=None):
        return get_forward_score(z_tensor, w_u[:, 0], b_u[:, 0], w_l[:, 0], b_l[:, 0], y_tensor, target_tensor)

    if IBP and forward:
        _, z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:8]
    if not IBP and forward:
        _, z, w_u_f, b_u_f, w_l_f, b_l_f = output[:6]
    if IBP and not forward:
        _, z, u_c, l_c = output[:4]

    if IBP:
        adv_ibp = get_ibp_score(u_c, l_c, source_labels, target_labels)
    if forward:
        adv_f = get_forward_score(z, w_u_f, b_u_f, w_l_f, b_l_f, source_labels, target_labels)

    if IBP and not forward:
        adv_score = adv_ibp
    if IBP and forward:
        adv_score = np.minimum(adv_ibp, adv_f)
    if not IBP and forward:
        adv_score = adv_f

    if mode == "backward":
        w_u_b, b_u_b, w_l_b, b_l_b = output[-4:]
        adv_b = get_backward_score(z, w_u_b, b_u_b, w_l_b, b_l_b, source_labels)
        adv_score = np.minimum(adv_score, adv_b)

    return adv_score


def check_adv_box(model, x_min, x_max, source_labels, target_labels=None, batch_size=-1, fast=True):
    """
    if the constant is positive, then it is a formal guarantee that there is an adversarial examples
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param source_labels: the list of label that should be predicted all the time in the box (either an integer, either an array that can contain multiple source labels for each sample)
    :param target_labels:the list of label that should never be predicted in the box (either an integer, either an array that can contain multiple target labels for each sample)
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: numpy array, vector with upper bounds for adversarial attacks
    """

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
    input_dim = np.prod(input_shape)
    x_ = x_min + 0 * x_min
    x_ = x_.reshape([-1] + input_shape)

    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)
    if isinstance(source_labels, int) or isinstance(source_labels, np.int64):
        source_labels = np.zeros((len(x_), 1)) + source_labels

    if isinstance(source_labels, list):
        source_labels = np.array(source_labels).reshape((len(x_), -1))

    source_labels = source_labels.reshape((len(x_), -1))
    source_labels = source_labels.astype("int64")

    if target_labels is not None:
        target_labels = target_labels.reshape((len(x_), -1))
        target_labels = target_labels.astype("int64")

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

        results = [check_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1, fast=fast) for i in range(len(X_min_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, z])

    def get_ibp_score(u_c, l_c, source_tensor, target_tensor=None):

        if target_tensor is None:
            target_tensor = 1 - source_tensor

        shape = np.prod(u_c.shape[1:])
        u_c_ = np.reshape(u_c, (-1, shape))
        l_c_ = np.reshape(l_c, (-1, shape))

        score_u = (
            l_c_ * target_tensor - np.expand_dims(np.min(u_c_ * source_tensor, -1), -1) - 1e6 * (1 - target_tensor)
        )

        return np.max(score_u, -1)

    def get_forward_score(z_tensor, w_u, b_u, w_l, b_l, source_tensor, target_tensor=None):

        if target_tensor is None:
            target_tensor = 1 - source_tensor

        n_dim = w_u.shape[1]
        shape = np.prod(b_u.shape[1:])
        w_u_ = np.reshape(w_u, (-1, n_dim, shape, 1))
        w_l_ = np.reshape(w_l, (-1, n_dim, 1, shape))
        b_u_ = np.reshape(b_u, (-1, shape, 1))
        b_l_ = np.reshape(b_l, (-1, 1, shape))

        w_u_f = w_l_ - w_u_
        b_u_f = b_l_ - b_u_

        # add penalties on biases
        b_u_f = b_u_f - 1e6 * (1 - target_tensor)[:, None, :]
        b_u_f = b_u_f - 1e6 * (1 - source_tensor)[:, :, None]

        upper = (
            np.sum(np.maximum(w_u_f, 0) * z_tensor[:, 1, :, None, None], 1)
            + np.sum(np.minimum(w_u_f, 0) * z_tensor[:, 0, :, None, None], 1)
            + b_u_f
        )

        return np.max(upper, (-1, -2))

    def get_backward_score(z_tensor, w_u, b_u, w_l, b_l, y_tensor, target_tensor=None):
        return get_forward_score(z_tensor, w_u[:, 0], b_u[:, 0], w_l[:, 0], b_l[:, 0], y_tensor, target_tensor)

    if IBP and forward:
        _, z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:8]
    if not IBP and forward:
        _, z, w_u_f, b_u_f, w_l_f, b_l_f = output[:6]
    if IBP and not forward:
        _, z, u_c, l_c = output[:4]

    if IBP:
        adv_ibp = get_ibp_score(u_c, l_c, source_labels, target_labels)
    if forward:
        adv_f = get_forward_score(z, w_u_f, b_u_f, w_l_f, b_l_f, source_labels, target_labels)

    if IBP and not forward:
        adv_score = adv_ibp
    if IBP and forward:
        adv_score = np.maximum(adv_ibp, adv_f)
    if not IBP and forward:
        adv_score = adv_f

    if mode == "backward":
        w_u_b, b_u_b, w_l_b, b_l_b = output[-4:]
        adv_b = get_backward_score(z, w_u_b, b_u_b, w_l_b, b_l_b, source_labels)
        adv_score = np.maximum(adv_score, adv_b)

    return adv_score

    """
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
    input_dim = np.prod(input_shape)
    x_ = x_min + 0 * x_min
    x_ = x_.reshape([-1] + input_shape)
    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)

    if isinstance(source_labels, int):
        source_labels = np.zeros((len(x_), 1)) + source_labels

    if isinstance(source_labels, list):
        source_labels = np.array(source_labels).reshape((len(x_), -1))

    source_labels = source_labels.reshape((len(x_), -1))
    source_labels = source_labels.astype("int64")

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

        results = [get_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1, fast=fast) for i in range(len(X_min_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, z])

    n_label = source_labels.shape[-1]

    if mode == Backward.name:

        w_u_b, b_u_b, w_l_b, b_l_b = output[-4:]
        w_u_b = w_u_b[:, 0]
        b_u_b = b_u_b[:, 0]
        w_l_b = w_l_b[:, 0]
        b_l_b = b_l_b[:, 0]

        for i in range(n_label):
            w_l_b = w_l_b - w_u_b[np.arange(len(w_l_b)), :, source_labels[:, i]][:, :, None]
            b_l_b = b_l_b - b_u_b[np.arange(len(b_l_b)), source_labels[:, i]][:, None]

        u_b = (
            np.sum(np.maximum(w_l_b, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_l_b, 0) * x_min[:, 0, :, None], 1)
            + b_l_b
        )

    if forward:
        if not IBP:
            _, _, w_u_f, b_u_f, w_l_f, b_l_f = output[:6]
        else:
            _, _, _, w_u_f, b_u_f, _, w_l_f, b_l_f = output[:8]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (w_u_f.shape[0], w_u_f.shape[1], -1))
            w_l_f = np.reshape(w_l_f, (w_l_f.shape[0], w_l_f.shape[1], -1))
            b_u_f = np.reshape(b_u_f, (len(b_u_f), -1))
            b_l_f = np.reshape(b_l_f, (len(b_l_f), -1))

        for i in range(n_label):
            w_l_f = w_l_f - w_u_f[np.arange(len(w_l_f)), :, source_labels[:, i]][:, :, None]
            b_l_f = b_l_f - b_u_f[np.arange(len(b_l_f)), source_labels[:, i]][:, None]

        u_f = (
            np.sum(np.maximum(w_l_f, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_l_f, 0) * x_min[:, 0, :, None], 1)
            + b_l_f
        )

        if IBP:
            u_i = output[2]
            l_i = output[5]

            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (len(u_i), -1))
                l_i = np.reshape(l_i, (len(l_i), -1))
            for i in range(n_label):
                l_i = l_i - u_i[np.arange(len(u_i)), source_labels[:, i]][:, None]

    else:
        u_i = output[2]
        l_i = output[3]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (len(u_i), -1))
            l_i = np.reshape(l_i, (len(l_i), -1))
            ######
        for i in range(n_label):
            l_i = l_i - u_i[np.arange(len(u_i)), source_labels[:, i]][:, None]

    if IBP and forward:
        u_ = np.minimum(l_i, u_f)
    if IBP and not forward:
        u_ = l_i
    if not IBP and forward:
        u_ = u_f

    if mode == Backward.name:
        u_ = np.minimum(u_, u_b)

    for i in range(n_label):
        u_[np.arange(len(u_)), source_labels[:, i]] = -np.inf

    if not fast:
        if mode == Backward.name and forward:
            u_ = -convex_optimize(-u_, -w_l_f, -b_l_f, -w_l_b, -b_l_b, z)
    #####
    if target_labels is None:
        return np.max(u_, -1)

    if isinstance(target_labels, int):
        target_labels = np.zeros((len(x_), 1)) + target_labels

    if isinstance(target_labels, list):
        target_labels = np.array(target_labels).reshape((len(x_), -1))

    target_labels = target_labels.reshape((len(x_), -1))
    target_labels = target_labels.astype("int64")

    n_target = target_labels.shape[-1]
    upper = u_[np.arange(len(u_)), target_labels[:, 0]]
    for i in range(n_target - 1):
        upper = np.maximum(upper, u_[np.arange(len(u_)), target_labels[:, i + 1]])

    return upper
    """


#### FORMAL BOUNDS ######


def get_upper_box(model, x_min, x_max, batch_size=-1, fast=True):
    """
    upper bound the maximum of a model in a given box
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: numpy array, vector with upper bounds for adversarial attacks
    """
    fast = True
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
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

        results = [get_upper_box(model_, X_min_[i], X_max_[i], -1, fast=fast) for i in range(len(X_min_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, z])

    if mode == Backward.name:

        w_u_b, b_u_b, _, _ = output[-4:]
        w_u_b = w_u_b[:, 0]
        b_u_b = b_u_b[:, 0]

        u_b = (
            np.sum(np.maximum(w_u_b, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_u_b, 0) * x_min[:, 0, :, None], 1)
            + b_u_b
        )
    if forward:
        if not IBP:
            _, _, w_u_f, b_u_f, _, _ = output[:6]
        else:
            _, _, _, w_u_f, b_u_f, _, _, _ = output[:8]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (w_u_f.shape[0], w_u_f.shape[1], -1))
            b_u_f = np.reshape(b_u_f, (len(b_u_f), -1))

        u_f = (
            np.sum(np.maximum(w_u_f, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_u_f, 0) * x_min[:, 0, :, None], 1)
            + b_u_f
        )

        if IBP:
            u_i = output[2]

            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (len(u_i), -1))

    else:
        u_i = output[2]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (len(u_i), -1))
            ######

    if IBP and forward:
        u_ = np.minimum(u_i, u_f)
    if IBP and not forward:
        u_ = u_i
    if not IBP and forward:
        u_ = u_f

    if mode == Backward.name:
        u_ = np.minimum(u_, u_b)

    return u_


def get_lower_box(model, x_min, x_max, batch_size=-1, fast=True):
    """
    lower bound the minimum of a model in a given box
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: numpy array, vector with lower bounds for adversarial attacks
    """
    fast = True
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
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

        results = [get_lower_box(model_, X_min_[i], X_max_[i], -1, fast=fast) for i in range(len(X_min_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, z])

    if mode == Backward.name:

        _, _, w_l_b, b_l_b = output[-4:]
        w_l_b = w_l_b[:, 0]
        b_l_b = b_l_b[:, 0]

        l_b = (
            np.sum(np.maximum(w_l_b, 0) * x_min[:, 0, :, None], 1)
            + np.sum(np.minimum(w_l_b, 0) * x_max[:, 0, :, None], 1)
            + b_l_b
        )

    if forward:
        if not IBP:
            _, _, _, _, w_l_f, b_l_f = output[:6]
        else:
            _, _, _, _, _, _, w_l_f, b_l_f = output[:8]

        # reshape if necessary
        if len(w_l_f.shape) > 3:
            w_l_f = np.reshape(w_l_f, (w_l_f.shape[0], w_l_f.shape[1], -1))
            b_l_f = np.reshape(b_l_f, (len(b_l_f), -1))

        l_f = (
            np.sum(np.maximum(w_l_f, 0) * x_min[:, 0, :, None], 1)
            + np.sum(np.minimum(w_l_f, 0) * x_max[:, 0, :, None], 1)
            + b_l_f
        )

        if IBP:
            l_i = output[5]

            if len(l_i.shape) > 2:
                l_i = np.reshape(l_i, (len(l_i), -1))

    else:
        l_i = output[3]
        if len(l_i.shape) > 2:
            l_i = np.reshape(l_i, (len(l_i), -1))
            ######

    if IBP and forward:
        l_ = np.maximum(l_i, l_f)
    if IBP and not forward:
        l_ = l_i
    if not IBP and forward:
        l_ = l_f

    if mode == Backward.name:
        l_ = np.maximum(l_, l_b)

    return l_


def get_range_box(model, x_min, x_max, batch_size=-1, fast=True):
    """
    bounding the outputs of a model in a given box
    if the constant is negative, then it is a formal guarantee that there is no adversarial examples
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: 2 numpy array, vector with upper bounds and vector with lower bounds
    """
    fast = True
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
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
        results = [get_range_box(model_, X_min_[i], X_max_[i], -1, fast=fast) for i in range(len(X_min_))]

        u_ = [r[0] for r in results]
        l_ = [r[1] for r in results]

        return np.concatenate(u_), np.concatenate(l_)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, z])

    if mode == Backward.name:

        w_u_b, b_u_b, w_l_b, b_l_b = output[-4:]
        w_u_b = w_u_b[:, 0]
        b_u_b = b_u_b[:, 0]
        w_l_b = w_l_b[:, 0]
        b_l_b = b_l_b[:, 0]

        u_b = (
            np.sum(np.maximum(w_u_b, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_u_b, 0) * x_min[:, 0, :, None], 1)
            + b_u_b
        )

        l_b = (
            np.sum(np.maximum(w_l_b, 0) * x_min[:, 0, :, None], 1)
            + np.sum(np.minimum(w_l_b, 0) * x_max[:, 0, :, None], 1)
            + b_l_b
        )

    if forward:
        if not IBP:
            _, _, w_u_f, b_u_f, w_l_f, b_l_f = output[:6]
        else:
            _, _, _, w_u_f, b_u_f, _, w_l_f, b_l_f = output[:8]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (w_u_f.shape[0], w_u_f.shape[1], -1))
            w_l_f = np.reshape(w_l_f, (w_l_f.shape[0], w_l_f.shape[1], -1))
            b_u_f = np.reshape(b_u_f, (len(b_u_f), -1))
            b_l_f = np.reshape(b_l_f, (len(b_l_f), -1))

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

        if IBP:
            u_i = output[2]
            l_i = output[5]

            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (len(u_i), -1))
                l_i = np.reshape(l_i, (len(l_i), -1))

    else:
        u_i = output[2]
        l_i = output[3]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (len(u_i), -1))
            l_i = np.reshape(l_i, (len(l_i), -1))
            ######

    if IBP and forward:
        u_ = np.minimum(u_i, u_f)
        l_ = np.maximum(l_i, l_f)
    if IBP and not forward:
        u_ = u_i
        l_ = l_i
    if not IBP and forward:
        u_ = u_f
        l_ = l_f

    if mode == Backward.name:
        u_ = np.minimum(u_, u_b)
        l_ = np.maximum(l_, l_b)

    #####
    return u_, l_


# get upper bound of a sample with bounded noise
def get_upper_noise(model, x, eps=-1, p=np.inf, batch_size=-1, fast=True):
    """
    upper bound the maximum of a model in an Lp Ball
    :param model: either a Keras model or a Decomon model
    :param x: numpy array, the example around
    which the impact of noise is assessed
    :param eps: the radius of the ball
    :param p: the type of Lp norm (p=2, 1, np.inf)
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: numpy array, vector with upper bounds
     of the range of values taken by the model inside the ball
    """
    # check that the model is a DecomonModel, else do the conversion
    convex_domain = {"name": Ball.name, "p": p, "eps": max(0, eps)}

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward", convex_domain=convex_domain)
    else:
        model_ = model
        if eps >= 0:
            model_.set_domain(convex_domain)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
    # input_dim = np.prod(input_shape)
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_ = [x_[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        results = [get_upper_noise(model_, X_[i], eps=eps, p=p, batch_size=-1, fast=fast) for i in range(len(X_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, x_.reshape((len(x_), -1))])

    x_ = x_.reshape((len(x_), -1))
    if p not in [1, 2, np.inf]:
        raise NotImplementedError()

    if p == np.inf:
        ord = 1
    if p == 1:
        ord = np.inf
    if p == 2:
        ord = 2

    if mode == Backward.name:

        w_u_b, b_u_b, _, _ = output[-4:]
        w_u_b = w_u_b[:, 0]
        b_u_b = b_u_b[:, 0]
        u_b = eps * np.linalg.norm(w_u_b, ord=ord, axis=1) + np.sum(w_u_b * x_[:, :, None], 1) + b_u_b

    if forward:
        if not IBP:
            _, _, w_u_f, b_u_f, _, _ = output[:6]
        else:
            _, _, _, w_u_f, b_u_f, _, _, _ = output[:8]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (w_u_f.shape[0], w_u_f.shape[1], -1))
            b_u_f = np.reshape(b_u_f, (len(b_u_f), -1))

        u_f = eps * np.linalg.norm(w_u_f, ord=ord, axis=1) + np.sum(w_u_f * x_[:, :, None], 1) + b_u_f

        if IBP:
            u_i = output[2]

            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (len(u_i), -1))

    else:
        u_i = output[2]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (len(u_i), -1))
            ######

    if IBP and forward:
        u_ = np.minimum(u_i, u_f)
    if IBP and not forward:
        u_ = u_i
    if not IBP and forward:
        u_ = u_f

    if mode == Backward.name:
        u_ = np.minimum(u_, u_b)

    return u_


# get upper bound of a sample with bounded noise
def get_lower_noise(model, x, eps, p=np.inf, batch_size=-1, fast=True):
    """
    lower bound the minimum of a model in an Lp Ball
    :param model: either a Keras model or a Decomon model
    :param x: numpy array, the example around
    which the impact of noise is assessed
    :param eps: the radius of the ball
    :param p: the type of Lp norm (p=2, 1, np.inf)
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: numpy array, vector with lower bounds
     of the range of values taken by the model inside the ball
    """
    # check that the model is a DecomonModel, else do the conversion
    convex_domain = {"name": Ball.name, "p": p, "eps": eps}

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward", convex_domain=convex_domain)
    else:
        model_ = model
        model_.set_domain(convex_domain)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
    input_dim = np.prod(input_shape)
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_ = [x_[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        results = [get_lower_noise(model_, X_[i], eps=eps, p=p, batch_size=-1, fast=fast) for i in range(len(X_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, x_.reshape((len(x_), -1))])

    x_ = x_.reshape((len(x_), -1))
    if p not in [1, 2, np.inf]:
        raise NotImplementedError()

    if p == np.inf:
        ord = 1
    if p == 1:
        ord = np.inf
    if p == 2:
        ord = 2

    if mode == Backward.name:

        _, _, w_l_b, b_l_b = output[-4:]
        w_l_b = w_l_b[:, 0]
        b_l_b = b_l_b[:, 0]

        l_b = -eps * np.linalg.norm(w_l_b, ord=ord, axis=1) + np.sum(w_l_b * x_[:, :, None], 1) + b_l_b

    if forward:
        if not IBP:
            _, _, _, _, w_l_f, b_l_f = output[:6]
        else:
            _, _, _, _, _, _, w_l_f, b_l_f = output[:8]

        # reshape if necessary
        if len(w_l_f.shape) > 3:
            w_l_f = np.reshape(w_l_f, (w_l_f.shape[0], w_l_f.shape[1], -1))
            b_l_f = np.reshape(b_l_f, (len(b_l_f), -1))

        l_f = -eps * np.linalg.norm(w_l_f, ord=ord, axis=-1) + np.sum(w_l_f * x_[:, 0, :, None], 1) + b_l_f

        if IBP:
            l_i = output[3]

            if len(l_i.shape) > 2:
                l_i = np.reshape(l_i, (len(l_i), -1))

    else:
        l_i = output[3]
        if len(l_i.shape) > 2:
            l_i = np.reshape(l_i, (len(l_i), -1))
            ######

    if IBP and forward:
        l_ = np.maximum(l_i, l_f)
    if IBP and not forward:
        l_ = l_i
    if not IBP and forward:
        l_ = l_f

    if mode == Backward.name:
        l_ = np.maximum(l_, l_b)

    return l_


# get upper bound of a sample with bounded noise
def get_range_noise(model, x, eps, p=np.inf, batch_size=-1, fast=True):
    """
    Bounds the output of a model in an Lp Ball
    :param model: either a Keras model or a Decomon model
    :param x: numpy array, the example around
    which the impact of noise is assessed
    :param eps: the radius of the ball
    :param p: the type of Lp norm (p=2, 1, np.inf)
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: 2 numpy arrays, vector with upper andlower bounds
    of the range of values taken by the model inside the ball
    """

    # check that the model is a DecomonModel, else do the conversion
    convex_domain = {"name": Ball.name, "p": p, "eps": eps}

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward", convex_domain=convex_domain)
    else:
        model_ = model
        model_.set_domain(convex_domain)

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
    # input_dim = np.prod(input_shape)
    x_ = x + 0 * x
    x_ = x_.reshape([-1] + input_shape)

    if batch_size > 0:
        # split
        r = 0
        if len(x_) % batch_size > 0:
            r += 1
        X_ = [x_[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        results = [get_range_noise(model_, X_[i], eps=eps, p=p, batch_size=-1, fast=fast) for i in range(len(X_))]

        u_ = [r[0] for r in results]
        l_ = [r[1] for r in results]

        return np.concatenate(u_), np.concatenate(l_)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, x_.reshape((len(x_), -1))])

    x_ = x_.reshape((len(x_), -1))
    if p not in [1, 2, np.inf]:
        raise NotImplementedError()

    if p == np.inf:
        ord = 1
    if p == 1:
        ord = np.inf
    if p == 2:
        ord = 2

    if mode == Backward.name:

        w_u_b, b_u_b, w_l_b, b_l_b = output[-4:]
        w_u_b = w_u_b[:, 0]
        b_u_b = b_u_b[:, 0]
        w_l_b = w_l_b[:, 0]
        b_l_b = b_l_b[:, 0]

        u_b = eps * np.linalg.norm(w_u_b, ord=ord, axis=1) + np.sum(w_u_b * x_[:, :, None], 1) + b_u_b
        l_b = -eps * np.linalg.norm(w_l_b, ord=ord, axis=1) + np.sum(w_l_b * x_[:, :, None], 1) + b_l_b

    if forward:
        if not IBP:
            _, _, w_u_f, b_u_f, w_l_f, b_l_f = output[:6]
        else:
            _, _, _, w_u_f, b_u_f, _, w_l_f, b_l_f = output[:8]

        # reshape if necessary
        if len(w_u_f.shape) > 3:
            w_u_f = np.reshape(w_u_f, (w_u_f.shape[0], w_u_f.shape[1], -1))
            b_u_f = np.reshape(b_u_f, (len(b_u_f), -1))
            w_l_f = np.reshape(w_l_f, (w_l_f.shape[0], w_l_f.shape[1], -1))
            b_l_f = np.reshape(b_l_f, (len(b_l_f), -1))

        u_f = eps * np.linalg.norm(w_u_f, ord=ord, axis=-1) + np.sum(w_u_f * x_[:, 0, :, None], 1) + b_u_f
        l_f = -eps * np.linalg.norm(w_l_f, ord=ord, axis=-1) + np.sum(w_l_f * x_[:, 0, :, None], 1) + b_l_f

        if IBP:
            u_i = output[2]
            l_i = output[3]

            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (len(u_i), -1))
                l_i = np.reshape(l_i, (len(l_i), -1))

    else:
        u_i = output[2]
        l_i = output[3]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (len(u_i), -1))
            l_i = np.reshape(l_i, (len(l_i), -1))
            ######

    if IBP and forward:
        u_ = np.minimum(u_i, u_f)
        l_ = np.maximum(l_i, l_f)
    if IBP and not forward:
        u_ = u_i
        l_ = l_i
    if not IBP and forward:
        u_ = u_f
        l_ = l_f

    if mode == Backward.name:
        u_ = np.minimum(u_, u_b)
        l_ = np.maximum(l_, l_b)

    return u_, l_


def refine_box(func, model, x_min, x_max, n_split, source_labels=None, target_labels=None, batch_size=-1, random=True):

    if func.__name__ not in [
        elem.__name__ for elem in [get_upper_box, get_lower_box, get_adv_box, check_adv_box, get_range_box]
    ]:
        raise NotImplementedError()

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, IBP=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model_.input_shape[0][1:])
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

            results = func(model_, x_min=X_min_, x_max=X_max_, batch_size=batch_size)

            if func.__name__ == get_upper_box.__name__:
                return results.reshape((n_x, n_split, -1))
            if func.__name__ == get_lower_box.__name__:
                return results.reshape((n_x, n_split, -1))

        if func.__name__ in [elem.__name__ for elem in [get_adv_box, check_adv_box]]:

            results = func(
                model_,
                x_min=X_min_,
                x_max=X_max_,
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
