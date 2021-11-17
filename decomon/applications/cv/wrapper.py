from __future__ import absolute_import
from decomon.models import DecomonModel, convert, clone
import numpy as np
from decomon.layers.core import Ball, Box
from decomon.models.decomon_sequential import Backward, Forward

# check inputs


##### ADVERSARIAL ROBUSTTNESS #####
def get_adv_brightness(
    model, x, bright_min, bright_max, source_labels, target_labels=None, x_min=None, x_max=None, batch_size=-1
):

    if np.min(bright_max - bright_min) < 0:
        raise UserWarning("Inconsistency Error: bright_max < bright_min")

    # check that the model is a DecomonModel, else do the conversion
    # input_dim = 0
    if not isinstance(model, DecomonModel):
        model_ = clone(model, IBP=True, mode="backward", input_dim=1)
    else:
        assert len(model.convex_domain) == 0 or model.convex_domain["name"] == Box.name
        model_ = model

    # reshape x_min, x_max
    input_shape = list(model_.input_shape[0][1:])
    input_dim = 1

    x_ = x
    x_ = x_.reshape([-1] + input_shape)

    # to do: a case by case depending on the input; whether it is clone or convert with a warning message

    w_ = np.ones_like(x_)[:, None]
    bias_ = x_
    n = len(x_)

    try:
        _ = len(bright_min)
        if isinstance(bright_min, list):
            bright_min = np.array(bright_min)
        bright_min = np.reshape(bright_min, (n, 1))
    except TypeError:
        bright_min = bright_min * np.ones((n, 1))

    try:
        _ = len(bright_max)
        if isinstance(bright_max, list):
            bright_max = np.array(bright_max)
        bright_max = np.reshape(bright_max, (n, 1))
    except TypeError:
        bright_max = bright_max * np.ones((n, 1))

    if x_min is None:
        x_min = x
        x_min = x_min.reshape((len(x_), -1))
        x_min = x + bright_min
        x_min = np.reshape(x_min, x_.shape)
    if x_max is None:
        x_max = x
        x_max = x_max.reshape((len(x_), -1))
        x_max = x + bright_max
        x_max = np.reshape(x_max, x_.shape)

    z = np.zeros((n, 2, 1))
    z[:, 0] = bright_min
    z[:, 1] = bright_max

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
        B_min_ = [bright_min[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
        B_max_ = [bright_max[batch_size * i : batch_size * (i + 1)] for i in range(len(x_) // batch_size + r)]
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
            get_adv_brightness(model_, X_[i], B_min_[i], B_max_[i], S_[i], T_[i], X_min_[i], X_max_[i], -1)
            for i in range(len(X_min_))
        ]

        return np.concatenate(results)

    mode = model_.mode
    IBP = model_.IBP
    forward = model_.forward

    # detect clone_mode
    clone_mode = model_.inputs[1].shape[-1] == 1

    if clone_mode:
        # assume clone mode
        if IBP and forward:
            output = model_.predict([x_, z, x_max, w_, bias_, x_min, w_, bias_])
        elif forward:
            output = model_.predict([x_, z, w_, bias_, w_, bias_])
        else:
            output = model_.predict([x_, z, x_max, x_min])
    else:
        # assert model_.inputs[1].shape[-1]
        print(
            "Warning: you have built your decomon model with the classical convert method. Note that regarding geometric transformations, the clone method will be more efficient"
        )

        z_box = np.concatenate([x_min[:, None], x_max[:, None]], 1)
        z_box = np.reshape(z_box, (len(z_box), 2, -1))
        output = model_.predict([x_, z_box])

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

        # we need to backward until the brightness variable
        w_u_ = np.sum(w_u[:, 0], 1)[:, None]
        w_l_ = np.sum(w_l[:, 0], 1)[:, None]
        tmp = np.reshape(x_, (len(x_), -1, 1))
        b_u_ = b_u[:, 0] + np.sum(w_u[:, 0] * tmp, 1)
        b_l_ = b_l[:, 0] + np.sum(w_l[:, 0] * tmp, 1)

        return get_forward_score(z_tensor, w_u_, b_u_, w_l_, b_l_, y_tensor, target_tensor)

    if IBP and forward:
        _, z, u_c, w_u_f, b_u_f, l_c, w_l_f, b_l_f = output[:8]
    if not IBP and forward:
        _, z, w_u_f, b_u_f, w_l_f, b_l_f = output[:6]
    if IBP and not forward:
        _, z, u_c, l_c = output[:4]

    if IBP:
        adv_ibp = get_ibp_score(u_c, l_c, source_labels, target_labels)
    if forward:

        if not clone_mode:
            w_u_ = np.sum(w_u_f, 1)[:, None]
            w_l_ = np.sum(w_l_f, 1)[:, None]
            tmp = np.reshape(x_, (len(x_), -1, 1))
            b_u_ = b_u_f + np.sum(w_u_f * tmp, 1)
            b_l_ = b_l_f + np.sum(w_l_f * tmp, 1)

            w_u_f = w_u_
            w_l_f = w_l_
            b_u_f = b_u_
            b_l_f = b_l_
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
