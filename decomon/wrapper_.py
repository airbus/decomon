# Set of functions that make it easy to use the library in a black box manner
# No need to understand the technical aspects or to code
from .models import DecomonModel, convert
import numpy as np
from .layers.core import Ball, Box
from .models.decomon_sequential import Backward, Forward
import scipy.optimize as opt


def convex_optimize(lc, w_1, b_1, w_2, b_2, z):

    # flatten u, w_1 and b_1
    lc = lc.reshape((lc.shape[0], -1))
    w_1 = w_1.reshape((w_1.shape[0], w_1.shape[1], -1))
    b_1 = b_1.reshape((b_1.shape[0], -1))
    w_2 = w_2[:, 0]
    b_2 = b_2[:, 0]

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


# get upper bound in a box
def get_upper_box(model, x_min, x_max, mode=Backward.name, fast=True):
    """


    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extramal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :param mode: forward or backward
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
        mode = model.mode
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

    if fast or mode == Forward.name:
        return output[2]
    else:
        # fast optimization
        u = output[2]
        w_1 = output[3]
        b_1 = output[4]
        w_2 = output[-4]
        b_2 = output[-3]

        return -convex_optimize(-u, -w_1, -b_1, -w_2, -b_2, z)


# get lower box in a box
def get_lower_box(model, x_min, x_max, mode=Backward.name, fast=True):
    """

    :param model: either a Keras model or a Decomon model
    :param x_min: numpy array for the extramal lower corner of the boxes
    :param x_max: numpy array for the extremal upper corner of the boxes
    :return: numpy array, vector with upper bounds of the
    range of values taken by the model inside every boxes
    """

    ###########
    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, mode=mode)
        input_dim = np.prod(model.input_shape[1:])
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model
        mode = model.mode
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

    if fast or mode == Forward.name:
        return output[5]
    else:
        # fast optimization
        l = output[5]
        w_1 = output[6]
        b_1 = output[7]
        w_2 = output[-2]
        b_2 = output[-1]

        return convex_optimize(l, w_1, b_1, w_2, b_2, z)


# get interval in a box
def get_range_box(model, x_min, x_max, fast=True):

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, ibp=True, mode="backward")
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_mmin, x_max
    input_shape = list(model.input_shape[0][1:])
    input_dim = np.prod(input_shape)
    x_ = x_min + 0 * x_min
    x_ = x_.reshape([-1] + input_shape)
    x_min = x_min.reshape((-1, 1, input_dim))
    x_max = x_max.reshape((-1, 1, input_dim))

    z = np.concatenate([x_min, x_max], 1)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model.predict([x_, z])

    if not fast:
        raise NotImplementedError()

    if mode == Backward.name:

        w_u, b_u, w_l, b_l = output[-4:]
        w_u = w_u[:, 0]
        b_u = b_u[:, 0]
        w_l = w_l[:, 0]
        b_l = b_l[:, 0]

        u_ = (
            np.sum(np.maximum(w_u, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_u, 0) * x_min[:, 0, :, None], 1)
            + b_u
        )
        l_ = (
            np.sum(np.maximum(w_l, 0) * x_min[:, 0, :, None], 1)
            + np.sum(np.minimum(w_l, 0) * x_max[:, 0, :, None], 1)
            + b_l
        )

        if forward:
            if not IBP:
                _, _, w_u_f, b_u_f, w_l_f, b_l_f = output[:6]
            else:
                _, _, _, w_u_f, b_u_f, _, w_l_f, b_l_f = output[:8]
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
                upper = output[2]
                lower = output[5]
                u_f = np.minimum(u_f, upper)
                l_f = np.maximum(l_f, lower)

        else:
            u_f = output[2]
            l_f = output[3]

        u_ = np.minimum(u_, u_f)
        l_ = np.maximum(l_, l_f)

    else:
        if forward:
            if not IBP:
                _, _, w_u, b_u, w_l, b_l = output[:6]
            else:
                _, _, _, w_u, b_u, _, w_l, b_l = output[:8]
            u_ = (
                np.sum(np.maximum(w_u, 0) * x_max[:, 0, :, None], 1)
                + np.sum(np.minimum(w_u, 0) * x_min[:, 0, :, None], 1)
                + b_u
            )
            l_ = (
                np.sum(np.maximum(w_l, 0) * x_min[:, 0, :, None], 1)
                + np.sum(np.minimum(w_l, 0) * x_max[:, 0, :, None], 1)
                + b_l
            )
            if IBP:
                upper = output[2]
                lower = output[5]
                u_ = np.minimum(u_, upper)
                l_ = np.maximum(l_, lower)

        else:
            _, _, u_, l_ = output[:4]

    return u_, l_

    raise NotImplementedError()


def get_range_box_(model, x_min, x_max, mode=Backward.name, fast=True):
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
        model_ = convert(model, mode=mode)
        input_dim = np.prod(model.input_shape[1:])
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model
        mode = model_.mode
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

    if fast or mode == Forward.name:
        return output[2], output[5]
    else:
        # fast optimization
        u = output[2]
        w_1 = output[3]
        b_1 = output[4]
        w_2 = output[-4]
        b_2 = output[-3]

        upper = -convex_optimize(-u, -w_1, -b_1, -w_2, -b_2, z)

        # fast optimization
        lc = output[5]
        w_1 = output[6]
        b_1 = output[7]
        w_2 = output[-2]
        b_2 = output[-1]

        lower = convex_optimize(lc, w_1, b_1, w_2, b_2, z)

        return upper, lower


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


def get_adv_box(model, x_min, x_max, source_labels, target_labels=None, batch_size=-1, fast=True):

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = convert(model, ibp=True, mode="backward")
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

    if isinstance(source_labels, int) or str(source_labels.dtype)[:3] == "int":
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
        if target_labels is not None:
            T_ = [target_labels[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        else:
            T_ = [None] * (len(x_) // batch_size + r)

        # results=[]
        # for i in range(len(X_min_)):
        #    if i==103:
        #        import pdb; pdb.set_trace()
        #    results.append(get_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1, fast=fast))
        results = [get_adv_box(model_, X_min_[i], X_max_[i], S_[i], T_[i], -1, fast=fast) for i in range(len(X_min_))]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    output = model_.predict([x_, z])

    if not fast:
        raise NotImplementedError()

    n_label = source_labels.shape[-1]

    if mode == Backward.name:

        w_u, b_u, w_l, b_l = output[-4:]
        w_u = w_u[:, 0]
        b_u = b_u[:, 0]
        w_l = w_l[:, 0]
        b_l = b_l[:, 0]

        for i in range(n_label):
            w_u = w_u - w_l[np.arange(len(w_u)), :, source_labels[:, i]][:, :, None]
            b_u = b_u - b_l[np.arange(len(b_u)), source_labels[:, i]][:, None]
            b_u[np.arange(len(b_u)), source_labels[:, i]] = -np.inf

        u_ = (
            np.sum(np.maximum(w_u, 0) * x_max[:, 0, :, None], 1)
            + np.sum(np.minimum(w_u, 0) * x_min[:, 0, :, None], 1)
            + b_u
        )
        # l_ = np.sum(np.maximum(w_l, 0) * x_min[:, 0, :, None], 1) + np.sum(np.minimum(w_l, 0) * x_max[:, 0, :, None],1) + b_l

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
                b_u_f[np.arange(len(b_u_f)), source_labels[:, i]] = -np.inf

            u_f = (
                np.sum(np.maximum(w_u_f, 0) * x_max[:, 0, :, None], 1)
                + np.sum(np.minimum(w_u_f, 0) * x_min[:, 0, :, None], 1)
                + b_u_f
            )
            # l_f = np.sum(np.maximum(w_l_f, 0) * x_min[:, 0, :, None], 1) + np.sum(
            #    np.minimum(w_l_f, 0) * x_max[:, 0, :, None],
            #    1) + b_l_f
            if IBP:
                upper = output[2]
                lower = output[5]

                if len(upper.shape) > 2:
                    upper = np.reshape(upper, (len(upper), -1))
                    lower = np.reshape(lower, (len(lower), -1))
                for i in range(n_label):
                    upper = upper - lower[np.arange(len(upper)), source_labels[:, i]][:, None]
                    upper[np.arange(len(upper)), source_labels[:, i]] = -np.inf
                u_f = np.minimum(u_f, upper)

        else:
            u_f = output[2]
            l_f = output[3]
            if len(u_f.shape) > 2:
                u_f = np.reshape(u_f, (len(u_f), -1))
                l_f = np.reshape(l_f, (len(l_f), -1))
                ######
            for i in range(n_label):
                u_f = u_f - l_f[np.arange(len(u_f)), source_labels[:, i]][:, None]
                u_f[np.arange(len(u_f)), source_labels[:, i]] = -np.inf
        u_ = np.minimum(u_, u_f)

    else:
        if forward:
            if not IBP:
                _, _, w_u, b_u, w_l, b_l = output[:6]
            else:
                _, _, _, w_u, b_u, _, w_l, b_l = output[:8]

            # reshape if necessary
            if len(w_u.shape) > 3:
                w_u = np.reshape(w_u, (w_u.shape[0], w_u.shape[1], -1))
                w_l = np.reshape(w_l, (w_l.shape[0], w_l.shape[1], -1))
                b_u = np.reshape(b_u, (len(b_u), -1))
                b_l = np.reshape(b_l, (len(b_l), -1))

            for i in range(n_label):
                w_u = w_u - w_l[np.arange(len(w_u)), :, source_labels[:, i]][:, :, None]
                b_u = b_u - b_l[np.arange(len(b_u)), source_labels[:, i]][:, None]
                # b_u[np.arange(len(b_u)), source_labels[:, i]] = -np.inf

            u_ = (
                np.sum(np.maximum(w_u, 0) * x_max[:, 0, :, None], 1)
                + np.sum(np.minimum(w_u, 0) * x_min[:, 0, :, None], 1)
                + b_u
            )

            for i in range(n_label):
                u_[np.arange(len(u_)), source_labels[:, i]] = -np.inf

            if IBP:
                upper = output[2]
                lower = output[5]
                if len(upper.shape) > 2:
                    upper = np.reshape(upper, (len(upper), -1))
                    lower = np.reshape(lower, (len(lower), -1))
                    ######
                for i in range(n_label):
                    upper = upper - lower[np.arange(len(upper)), source_labels[:, i]][:, None]
                    upper[np.arange(len(upper)), source_labels[:, i]] = -np.inf

                u_ = np.minimum(u_, upper)

        else:
            _, _, u_, l_ = output[:4]
            if len(u_.shape) > 2:
                u_ = np.reshape(u_, (len(u_), -1))
                l_ = np.reshape(l_, (len(l_), -1))
            for i in range(n_label):
                u_ = u_ - l_[np.arange(len(u_)), source_labels[:, i]][:, None]
                u_[np.arange(len(u_)), source_labels[:, i]] = -np.inf
    #####
    if target_labels is None:
        return np.max(u_, -1)

    raise NotImplementedError()


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
