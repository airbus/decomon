from .models import DecomonModel, convert
import numpy as np
from .layers.core import Ball, Box
from .models.decomon_sequential import Backward, Forward
import scipy.optimize as opt


def convex_optimize(lc, w_1, b_1, w_2, b_2, z):

    # flatten u, w_1 and b_1
    w_1 = w_1.reshape((w_1.shape[0], w_1.shape[1], -1))
    b_1 = b_1.reshape((b_1.shape[0], -1))

    w_1_ = w_1.transpose((0, 2, 1)).reshape((-1, z.shape[-1]))
    w_2_ = w_2.transpose((0, 2, 1)).reshape((-1, z.shape[-1]))

    batch = len(z)
    x_0_ = np.expand_dims(0.5 * np.sum(z, 1), -1) + np.zeros_like(w_1)
    x_0 = x_0_.flatten()
    x_min = (np.expand_dims(z[:, 0], -1) + np.zeros_like(w_1)).flatten()
    x_max = (np.expand_dims(z[:, 1], -1) + np.zeros_like(w_1)).flatten()

    def func_(x_0):
        x_0_ = x_0.reshape((batch, z.shape[-1], -1))
        f_0 = lc
        f_1 = np.sum(w_1 * x_0_, 1) + b_1
        f_2 = np.sum(w_2 * x_0_, 1) + b_2

        f_ = np.maximum(np.maximum(f_0, f_1), f_2)
        return f_

    def func(x_0):
        f_ = np.sum(func_(x_0))
        return f_

    def grad(x_0):
        x_0_ = x_0.reshape((batch, z.shape[-1], -1))
        n_out = x_0_.shape[-1]
        f_0 = lc
        f_1 = np.sum(w_1 * x_0_, 1) + b_1
        f_2 = np.sum(w_2 * x_0_, 1) + b_2

        f_0 = f_0.flatten()  # (batch*n_out)
        f_1 = f_1.flatten()
        f_2 = f_2.flatten()

        index_0 = np.where(f_0 >= np.maximum(f_1, f_2))[0]
        index_1 = np.where(f_1 >= np.maximum(f_0, f_2))[0]
        index_2 = np.where(f_2 >= np.maximum(f_0, f_1))[0]

        grad_x_ = np.zeros((batch * n_out, z.shape[-1]))
        grad_x_[index_2] = w_2_[index_2]
        grad_x_[index_1] = w_1_[index_1]
        grad_x_[index_0] *= 0.0

        grad_x_ = (grad_x_.reshape((batch, n_out, z.shape[-1]))).transpose((0, 2, 1)).flatten()

        return grad_x_.astype(np.float64)

    result = opt.minimize(
        fun=func,
        jac=grad,
        x0=x_0,
        method="L-BFGS-B",
        bounds=opt.Bounds(x_min, x_max),
    )

    if result["success"]:
        return func_(result["x"])
    else:
        return lc.reshape((batch, z.shape[-1], -1))


##### ADVERSARIAL ROBUSTTNESS #####
def get_adv_box(model, x_min, x_max, source_labels, target_labels=None, batch_size=-1, fast=True):
    """
    if the constant is negative, then it is a formal guarantee that there is no adversarial examples
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
            w_u_b = w_u_b - w_l_b[np.arange(len(w_u_b)), :, source_labels[:, i]][:, :, None]
            b_u_b = b_u_b - b_l_b[np.arange(len(b_u_b)), source_labels[:, i]][:, None]

        u_b = (
            np.sum(np.maximum(w_u_b, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_u_b, 0) * x_min[:, 0, :, None], 1)
            + b_u_b
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
            w_u_f = w_u_f - w_l_f[np.arange(len(w_u_f)), :, source_labels[:, i]][:, :, None]
            b_u_f = b_u_f - b_l_f[np.arange(len(b_u_f)), source_labels[:, i]][:, None]

        u_f = (
            np.sum(np.maximum(w_u_f, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_u_f, 0) * x_min[:, 0, :, None], 1)
            + b_u_f
        )

        if IBP:
            u_i = output[2]
            l_i = output[5]

            if len(u_i.shape) > 2:
                u_i = np.reshape(u_i, (len(u_i), -1))
                l_i = np.reshape(l_i, (len(l_i), -1))
            for i in range(n_label):
                u_i = u_i - l_i[np.arange(len(u_i)), source_labels[:, i]][:, None]

    else:
        u_i = output[2]
        l_i = output[3]
        if len(u_i.shape) > 2:
            u_i = np.reshape(u_i, (len(u_i), -1))
            l_i = np.reshape(l_i, (len(l_i), -1))
            ######
        for i in range(n_label):
            u_i = u_i - l_i[np.arange(len(u_i)), source_labels[:, i]][:, None]

    if IBP and forward:
        u_ = np.minimum(u_i, u_f)
    if IBP and not forward:
        u_ = u_i
    if not IBP and forward:
        u_ = u_f

    if mode == Backward.name:
        u_ = np.minimum(u_, u_b)

    for i in range(n_label):
        u_[np.arange(len(u_)), source_labels[:, i]] = -np.inf

    if not fast:
        if mode == Backward.name and forward:
            u_ = -convex_optimize(-u_, -w_u_f, -b_u_f, -w_u_b, -b_u_b, z)
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


def check_adv_box(model, x_min, x_max, source_labels, target_labels=None, batch_size=-1, fast=True):
    """
    if the constant is positive, then it is a formal guarantee that there IS adversarial examples
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


#### FORMAL BOUNDS ######


def get_upper_box(model, x_min, x_max, batch_size=-1, fast=True):
    """
    if the constant is negative, then it is a formal guarantee that there is no adversarial examples
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
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

    if not fast:
        if mode == Backward.name and forward:
            u_ = -convex_optimize(-u_, -w_u_f, -b_u_f, -w_u_b, -b_u_b, z)

    return u_


def get_lower_box(model, x_min, x_max, batch_size=-1, fast=True):
    """
    if the constant is negative, then it is a formal guarantee that there is no adversarial examples
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: numpy array, vector with lower bounds for adversarial attacks
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

    if not fast:
        if mode == Backward.name and forward:
            l_ = convex_optimize(l_, w_l_f, b_l_f, w_l_b, b_l_b, z)

    return l_


def get_range_box(model, x_min, x_max, batch_size=-1, fast=True):
    """
    if the constant is negative, then it is a formal guarantee that there is no adversarial examples
    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extremal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param batch_size: for computational efficiency, one can split the calls to minibatches
    :param fast: useful in the forward-backward or in the hybrid-backward mode to optimize the scores
    :return: 2 numpy array, vector with upper bounds and vector with lower bounds
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

    if not fast:
        if mode == Backward.name and forward:
            u_ = -convex_optimize(-u_, -w_u_f, -b_u_f, -w_u_b, -b_u_b, z)
            l_ = convex_optimize(l_, w_l_f, b_l_f, w_l_b, b_l_b, z)
    #####
    return u_, l_


# get upper bound of a sample with bounded noise
def get_upper_noise(model, x, eps=-1, p=np.inf, batch_size=-1, fast=True):
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
    input_dim = np.prod(input_shape)
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
    if not p in [1, 2, np.inf]:
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
    if not p in [1, 2, np.inf]:
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
        results = [get_range_noise(model_, X_[i], eps=eps, p=p, batch_size=-1, fast=fast) for i in range(len(X_))]

        u_ = [r[0] for r in results]
        l_ = [r[1] for r in results]

        return np.concatenate(u_), np.concatenate(l_)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, x_.reshape((len(x_), -1))])

    x_ = x_.reshape((len(x_), -1))
    if not p in [1, 2, np.inf]:
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

    if not func.__name__ in [
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

    for n_i in range(n_x):
        count = 1

        while count < n_split:

            if not random:
                i = np.argmax(np.max(index_max[n_i, :count], -1))
                j = np.argmax(index_max[n_i, i])
            else:
                i = np.random.randint(count)
                j = np.random.randint(input_dim)

            z_min = X_min[n_i, i]
            z_max = X_max[n_i, i]
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

    if func.__name__ in [elem.__name__ for elem in [get_upper_box, get_lower_box, get_range_box]]:

        results = func(model_, x_min=X_min_, x_max=X_max_, batch_size=batch_size)

        if func.__name__ == get_upper_box.__name__:
            return np.max(results.reshape((n_x, n_split, -1)), 1)
        if func.__name__ == get_lower_box.__name__:
            return np.min(results.reshape((n_x, n_split, -1)), 1)
        u_, l_ = results
        u_ = u_.reshape((n_x, n_split, -1))
        l_ = l_.reshape((n_x, n_split, -1))
        return np.max(u_, 1), np.min(l_, 1)

    if func.__name__ in [elem.__name__ for elem in [get_adv_box, check_adv_box]]:

        results = func(
            model_,
            x_min=X_min_,
            x_max=X_max_,
            source_labels=source_labels,
            target_labels=target_labels,
            batch_size=batch_size,
        )

        return np.max(results.reshape((n_x, n_split)), 1)
