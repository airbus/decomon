# Set of functions that make it easy to use the library in a black box manner
# No need to understand the technical aspects or to code
from .models import DecomonModel, convert
import numpy as np
from .layers.core import Ball, Box


# get upper bound in a box
def get_upper_box(model, x_min, x_max):
    """


    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extramal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :return: numpy array, vector with upper bounds of
    the range of values taken by the model inside every boxes
    """

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model)
        input_dim = np.prod(model.input_shape[1:])
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model
        input_dim = np.prod(model.input_shape[0][1:])

    if len(x_min.shape) == 1:
        if x_min.shape[0] != input_dim:
            x_min = x_min[:, None]
        else:
            x_min = x_min[None]
    if len(x_max.shape) == 1:
        if x_max.shape[0] != input_dim:
            x_max = x_max[:, None]
        else:
            x_max = x_max[None]

    z = np.concatenate([x_min[:, None], x_max[:, None]], 1)
    output = model_.predict([x_min, z])

    return output[2]


# get lower box in a box
def get_lower_box(model, x_min, x_max):
    """

    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extramal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :return: numpy array, vector with upper bounds of the
    range of values taken by the model inside every boxes
    """
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model)
        input_dim = np.prod(model.input_shape[1:])
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model
        input_dim = np.prod(model.input_shape[0][1:])

    if len(x_min.shape) == 1:
        if x_min.shape[0] != input_dim:
            x_min = x_min[:, None]
        else:
            x_min = x_min[None]
    if len(x_max.shape) == 1:
        if x_max.shape[0] != input_dim:
            x_max = x_max[:, None]
        else:
            x_max = x_max[None]

    z = np.concatenate([x_min[:, None], x_max[:, None]], 1)
    output = model_.predict([x_min, z])

    return output[5]


# get interval in a box
def get_range_box(model, x_min, x_max):
    """

    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extramal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :return: numpy array, 2D vector with upper bounds and lower bounds
     of the range of values taken by the model inside the box
    """
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model)
        input_dim = np.prod(model.input_shape[1:])
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model
        input_dim = np.prod(model.input_shape[0][1:])

    if len(x_min.shape) == 1:
        if x_min.shape[0] != input_dim:
            x_min = x_min[:, None]
        else:
            x_min = x_min[None]
    if len(x_max.shape) == 1:
        if x_max.shape[0] != input_dim:
            x_max = x_max[:, None]
        else:
            x_max = x_max[None]

    z = np.concatenate([x_min[:, None], x_max[:, None]], 1)
    output = model_.predict([x_min, z])

    return output[2], output[5]


# get upper bound of a sample with bounded noise
def get_upper_noise(model, x, eps, p=np.inf):
    """

    :param model: either a Keras model or a Decomon model
    :param x: numpy array, the example around
    which the impact of noise is assessed
    :param eps:
    :param p:
    :return: numpy array, vector with upper bounds
     of the range of values taken by the model inside the box
    """
    # check that the model is a DecomonModel, else do the conversion
    convex_domain = {"name": Ball.name, "p": p, "eps": eps}

    if not isinstance(model, DecomonModel):
        model_ = convert(model, convex_domain=convex_domain)
    else:
        model_ = model
        model_.set_domain(convex_domain)

    output = model_.predict([x, x])
    return output[2]


# get lower bound of a sample with bounded noise
def get_lower_noise(model, x, eps, p=np.inf):
    """

    :param model: either a Keras model or a Decomon model
    :param x: numpy array, the example around
    which the impact of noise is assessed
    :param eps: float, the magnitude of the noise
    :param p: Lp norm
    :return: numpy array, 2D vector with lower bounds
     of the range of values taken by the model inside the box
    """
    # check that the model is a DecomonModel, else do the conversion
    convex_domain = {"name": Ball.name, "p": p, "eps": eps}

    if not isinstance(model, DecomonModel):
        model_ = convert(model, convex_domain=convex_domain)
    else:
        model_ = model
        model_.set_domain(convex_domain)

    output = model_.predict([x, x])
    return output[5]


# get range bound of a sample with bounded noise
def get_range_noise(model, x, eps, p=np.inf):
    """

    :param model: either a Keras model or a Decomon model
    :param x: numpy array, the example around
    which the impact of noise is assessed
    :param eps: numpy array, the example around
    which the impact of noise is assessed
    :param p: Lp norm
    :return: numpy array, 2D vector with upper bounds and lower bounds
     of the range of values taken by the model inside the box
    """
    convex_domain = {"name": Ball.name, "p": p, "eps": eps}

    if not isinstance(model, DecomonModel):
        model_ = convert(model, convex_domain=convex_domain)
    else:
        model_ = model
        model_.set_domain(convex_domain)

    output = model_.predict([x, x])
    return output[2], output[5]


def get_adv_box(model, x_min, x_max, source_label, target_labels=None, fast=True):
    """

    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extramal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param source_label: the list of label
    that should be predicted all the time in the box
    :param target_labels: the label against which we want
    to assess robustness. If set to None, check for every other labels
    than source_label
    :param fast: boolean to have faster but less tight results
    :return: numpy array, vector with upper bounds for adversarial attacks
    """
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    if len(x_min.shape) == 1:
        x_min = x_min[None]
    if len(x_max.shape) == 1:
        x_max = x_max[None]

    if not isinstance(source_label, int):
        source_label = np.array(source_label)

    z = np.concatenate([x_min[:, None], x_max[:, None]], 1)
    output = model_.predict([x_min, z])

    u_, w_u, b_u, l_, w_l, b_l = output[2:8]

    if isinstance(source_label, int) or len(source_label.shape) == 0:
        upper = u_ - l_[:, source_label : source_label + 1]
        upper[:, source_label] = -np.inf
    else:
        upper = u_ - l_[range(len(source_label)), source_label][:, None]
        upper[range(len(source_label)), source_label] = -np.inf

    if not fast:

        if isinstance(source_label, int) or len(source_label.shape) == 0:
            w_ = w_u - w_l[:, :, source_label : source_label + 1]
            b_ = b_u - b_l[:, source_label : source_label + 1]

        else:
            w_ = w_u - w_l[:, :, source_label]
            b_ = b_u - b_l[:, source_label]

        x_min_ = x_min
        x_max_ = x_max
        n_expand = len(w_.shape) - len(x_min_.shape)
        for _i in range(n_expand):
            x_min_ = np.expand_dims(x_min_, -1)
            x_max_ = np.expand_dims(x_max_, -1)

        upper_ = np.sum(np.maximum(w_, 0) * x_max_, 1) + np.sum(np.minimum(w_, 0) * x_min_, 1) + b_
        upper_[:, source_label] = -np.inf

        upper = np.minimum(upper, upper_)

    if target_labels is None:
        return np.max(upper, -1)
    else:
        if isinstance(target_labels, int):
            return np.max(upper[:, target_labels], -1)
        else:
            return np.max(upper[:, np.array(target_labels)], -1)


def check_adv_box(model, x_min, x_max, source_label, target_labels=None, fast=True):
    """

    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extramal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param source_label: the list of label that should
    be predicted all the time in the box
    :param target_labels: the label against which we want
    to assess robustness. If set to None, check for every other labels
    than source_label
    :param fast: boolean to have faster but less tight results
    :return: numpy array, vector with upper bounds for adversarial attacks
    """
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    if len(x_min.shape) == 1:
        x_min = x_min[None]
    if len(x_max.shape) == 1:
        x_max = x_max[None]

    if not isinstance(source_label, int):
        source_label = np.array(source_label)

    z = np.concatenate([x_min[:, None], x_max[:, None]], 1)
    output = model_.predict([x_min, z])

    """
    preds = output[0]
    if isinstance(source_label, int) or len(source_label.shape) == 0:
    preds_label = preds[:, source_label:source_label+1][:, 0]
    preds -= preds_label[:, None]
    preds[:, label] = -1

    if preds.max() > 0:
        return 1, 0.
    """
    u_, w_u, b_u, l_, w_l, b_l = output[2:8]

    if isinstance(source_label, int) or len(source_label.shape) == 0:
        lower = l_ - u_[:, source_label : source_label + 1]
        lower[:, source_label] = -np.inf
    else:
        lower = l_ - u_[range(len(source_label)), source_label][:, None]
        lower[range(len(source_label)), source_label] = -np.inf

    if not fast:

        if isinstance(source_label, int) or len(source_label.shape) == 0:
            w_ = w_l - w_u[:, :, source_label : source_label + 1]
            b_ = b_l - b_u[:, source_label : source_label + 1]

        else:
            w_ = w_l - w_u[:, :, source_label]
            b_ = b_l - b_u[:, source_label]

        x_min_ = x_min
        x_max_ = x_max
        n_expand = len(w_.shape) - len(x_min_.shape)
        for _i in range(n_expand):
            x_min_ = np.expand_dims(x_min_, -1)
            x_max_ = np.expand_dims(x_max_, -1)

        lower_ = np.sum(np.maximum(w_, 0) * x_max_, 1) + np.sum(np.minimum(w_, 0) * x_min_, 1) + b_
        lower_[:, source_label] = -np.inf

        lower = np.minimum(lower, lower_)

    if target_labels is None:
        return np.max(lower, -1)
    else:
        if isinstance(target_labels, int):
            return np.max(lower[:, target_labels], -1)
        else:
            return np.max(lower[:, np.array(target_labels)], -1)


def init_alpha(model, decomon_model, x_min, x_max, label, fast=False, n_sub=1000):
    """Sampling a robust box along the diagonal of x_min and x_max

    :param model: Keras Model
    :param decomon_model: DecomonModel
    :param x_min: the extremal lower corner of the box
    :param x_max: the extremal upper corner of the box
    :param label: assessing that the model always predict label
    :param fast: boolean to have faster but less tight results
    :param n_sub: the number of subdivisions computed along the diagonal
    :return: a tuple (integer, numpy array) that indicates
    whether we found a robust zone along the diagonal and this
    sample along the diagonal
    """
    # assert that there is no adversarial examples
    # assert there is not adversarial examples
    z = np.concatenate([x_min, x_max], 0)
    preds = model.predict(z)
    preds_label = preds[:, label]
    preds -= preds_label[:, None]
    preds[:, label] = -1

    if preds.max() > 0:
        return 1, 0.0

    # check that the whole domain is not working
    upper_high = get_adv_box(decomon_model, x_min, x_max, source_label=label, fast=fast)[0]
    if upper_high <= 0:
        return -1, 1.0

    # test in // multiple values of splitting along the diagonal
    alpha = np.linspace(0.0, 1.0, n_sub)[:, None]
    X_alpha = (1 - alpha) * x_min + alpha * x_max

    upper = get_adv_box(decomon_model, x_min + 0 * X_alpha, X_alpha, source_label=label, fast=fast)
    index = np.where(upper < 0)[0][-1]
    return -1, alpha[index]


def increase_alpha(decomon_model, x_min, x_alpha, x_max, label, fast=False, alpha_=1):
    """Improving the current safe box coordinate wise

    :param decomon_model: DecomonModel
    :param x_min: the extremal lower corner of the box
    :param x_max: the extremal upper corner of the box
    :param x_alpha: the extremal corner of a robust box [x_min, x_alpha]
    :param label: assessing that the model always predict label
    :param fast: boolean to have faster but less tight results
    :param alpha_: controlling the ratio between x_max
    and x_alpha to increase a component of x_alpha
    :return: a tuple (numpy array, bool) with the update
    (or not) value of x_alpha and a boolean to assess whether
    we asucceed in enlarging the box
    """
    n_dim = np.prod(x_min.shape)
    input_shape = list(x_min.shape)[1:]
    X_min = np.concatenate([x_min] * n_dim, 0)

    # flatten X_min and x_alpha
    x_alpha_ = x_alpha.reshape((-1, n_dim))
    X_min_ = X_min.reshape((-1, n_dim))
    X_min_[np.arange(n_dim), np.arange(n_dim)] = x_alpha_
    X_min = X_min_.reshape(tuple([-1] + input_shape))
    x_alpha = x_alpha_.reshape(tuple([-1] + input_shape))
    upper_low = get_adv_box(decomon_model, X_min, x_alpha + 0 * X_min, source_label=label, fast=fast)

    index_keep = np.array(
        [i for i in range(n_dim) if (i in np.where(upper_low < 0)[0]) and (i in np.where(x_alpha[0] < x_max[0])[0])]
    )
    score = np.min(upper_low)
    j = 0
    while score.min() < 0:

        x_alpha_ = x_alpha.reshape((-1, n_dim))
        A = np.concatenate([x_min] * n_dim, 0)
        A_ = A.reshape((-1, n_dim))
        A_[np.arange(n_dim), np.arange(n_dim)] = x_alpha_[0]
        A = A_.reshape(tuple([-1] + input_shape))

        X_alpha = np.concatenate([x_alpha] * n_dim, 0)
        X_alpha_ = X_alpha.reshape((-1, n_dim))
        X_alpha_[np.arange(n_dim), np.arange(n_dim)] = alpha_ * x_max.reshape((-1, n_dim)) + (
            1 - alpha_
        ) * x_alpha.reshape((-1, n_dim))
        X_alpha = X_alpha_.reshape(tuple([-1] + input_shape))

        toto = get_adv_box(decomon_model, A, x_max + 0 * A, label, fast=fast)

        upper_dir = get_adv_box(
            decomon_model,
            A[index_keep],
            X_alpha[index_keep],
            source_label=label,
            fast=fast,
        )
        score = x_max - x_alpha
        score[0, index_keep] = upper_dir * score[0, index_keep]

        upper_ = np.ones((784,))
        upper_[index_keep] = upper_dir

        # upper_[np.where(x_max[0]==x_alpha[0])[0]]= 1
        if upper_.min() >= 0:
            break
        index = upper_.argmin()
        x_alpha[0, index] = alpha_ * x_max[0, index] + (1 - alpha_) * x_alpha[0, index]
        tata = get_adv_box(decomon_model, x_min, x_max, source_label=label, fast=fast)
        if tata < toto.max():
            import pdb

            pdb.set_trace()
        print(upper_.min(), upper_.argmin(), toto.max())
        # index_keep = np.where(score[0] < 0 and x_alpha[0]<x_max[0])[0]
        index_keep = np.array(
            [i for i in range(n_dim) if (i in np.where(upper_dir < 0)[0]) and (i in np.where(x_alpha[0] < x_max[0])[0])]
        )
        j += 1

    x_alpha_ = x_alpha.reshape((-1, n_dim))
    A = np.concatenate([x_min] * n_dim, 0)
    A_ = A.reshape((-1, n_dim))
    A_[np.arange(n_dim), np.arange(n_dim)] = x_alpha_[0]
    A = A_.reshape(tuple([-1] + input_shape))
    index_keep = np.where(x_alpha[0] < x_max[0])[0]
    toto = get_adv_box(
        decomon_model,
        A[index_keep],
        x_alpha + 0 * A[index_keep],
        source_label=label,
        fast=True,
    )
    return x_alpha, toto.min() < 0


def search_space(
    model,
    x_min,
    x_max,
    label,
    decomon_model=None,
    fast=False,
    n_sub=1000,
    n_iter=4,
    n_step=2,
):
    """

    :param model: Keras model
    :param x_min: the extremal lower corner of the box
    :param x_max: the extremal upper corner of the box
    :param decomon_model: DecomonModel or None,
    if set to None the convertion is done automatically
    :param fast: boolean to have faster but less tight results
    :param n_sub: ??
    :param n_iter: ??
    :param n_step: number of allowed recursive calls
    :return:
    """

    if decomon_model is None:
        decomon_model = convert(model)

    # first test on the whole domain
    lower = check_adv_box(decomon_model, x_min, x_max, source_label=label, fast=fast)
    if lower >= 0:
        return 1  # we found an adversarial example
    upper = get_adv_box(decomon_model, x_min, x_max, source_label=label, fast=fast)
    print("--", upper)
    if upper <= 0:
        return -1  # robust box

    if n_step == 0:
        return 0  # no more budget and the evaluation
        # of the cell with decomon was not sufficient

    n_dim = np.prod(x_min.shape)
    input_shape = list(x_min.shape)[1:]

    # find the best granularity
    sanity_test_0, alpha_ = init_alpha(model, decomon_model, x_min, x_max, label)
    if sanity_test_0 == 1:
        return 1  # we found an adversarial example (either x_min or x_max)
    if alpha_ == 1.0:
        return -1  # robust box

    x_alpha = (1 - alpha_) * x_min + alpha_ * x_max

    found = True
    i = 0
    alpha_tmp = np.linspace(0.5, 1.0, 4)[::-1]
    while found and i < len(alpha_tmp):
        # print(i, alpha_tmp[i])
        x_alpha, found = increase_alpha(decomon_model, x_min, x_alpha, x_max, label, alpha_=alpha_tmp[i])
        if np.allclose(x_max, x_alpha):
            return -1
        break

    X_min = np.concatenate([x_min] * n_dim, 0)
    X_min_ = X_min.reshape((-1, n_dim))
    X_min_[np.arange(n_dim), np.arange(n_dim)] = x_alpha.reshape((-1, n_dim))
    X_min = X_min_.reshape(tuple([-1] + input_shape))
    z = np.concatenate([X_min, x_alpha], 0)
    preds = model.predict(z)
    preds_label = preds[:, label]
    preds -= preds_label[:, None]
    preds[:, label] = -1

    if preds.max() > 0:
        return 1

    # work sequentially (not optimal but ok for now)
    for i in range(n_dim):
        if x_alpha[0, i] == x_max[0, i]:
            continue
        sanity_test_i = search_space(
            model,
            X_min[i : i + 1],
            x_max,
            label,
            decomon_model,
            fast,
            n_sub,
            n_iter,
            n_step - 1,
        )
        if sanity_test_i == 1:
            return 1  # we found an adversarial example
        if sanity_test_i == 0:
            return 0  # undetermined region

    return -1  # we validate every subdomains
