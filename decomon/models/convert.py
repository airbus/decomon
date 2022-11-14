from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import Flatten, Input, InputLayer, Lambda, Layer
from tensorflow.keras.models import Model

from decomon.models.backward_cloning import get_backward_model as convert_backward
from decomon.models.forward_cloning import convert_forward
from decomon.utils import Ball

from ..layers.decomon_layers import to_monotonic
from ..layers.utils import get_lower, get_upper
from ..numpy.models.convert import clone as clone_to_numpy
from .models import DecomonModel
from .utils import convert_2_mode, get_mode


class FORWARD_FEED:
    name = "feed_forward"


class BACKWARD_FEED:
    name = "feed_backward"


# define the set of methods
class CROWN:
    name = "crown"


class CROWN_IBP:
    name = "crown-ibp"


class CROWN_FORWARD:
    name = "crown-forward"


class CROWN_HYBRID:
    name = "crown-hybrid"


class IBP:
    name = "ibp"


class FORWARD:
    name = "forward"


class HYBRID:
    name = "hybrid"


def get_direction(method):
    if method in [IBP.name, FORWARD.name, HYBRID.name]:
        return FORWARD_FEED.name
    else:
        return BACKWARD_FEED.name


def get_ibp_forward_from_method(method):
    if method in [IBP.name, CROWN_IBP.name]:
        return True, False
    if method in [FORWARD.name, CROWN_FORWARD.name]:
        return False, True
    if method in [HYBRID.name, CROWN_HYBRID.name]:
        return True, True
    if method == CROWN.name:
        return False, True
    return True, True


def switch_mode_mapping(forward_map, IBP, forward, method):
    raise NotImplementedError()


# create status
def convert(
    model,
    input_tensors,
    method=CROWN.name,
    ibp=False,
    forward=False,
    back_bounds=[],
    layer_fn=to_monotonic,
    input_dim=-1,
    convex_domain={},
    finetune=False,
    shared=True,
    softmax_to_linear=True,
    layer_map={},
    forward_map={},
    finetune_forward=False,
    finetune_backward=False,
    final_ibp=False,
    final_forward=False,
    **kwargs,
):

    if finetune:
        finetune_forward = True
        finetune_backward = True

    method = method.lower()
    if not method in [algo.name for algo in [CROWN, CROWN_FORWARD, CROWN_HYBRID, CROWN_IBP, IBP, FORWARD, HYBRID]]:
        raise KeyError()

    if method != CROWN.name:

        # ibp_, forward_ = get_ibp_forward_from_method(method)
        ibp_, forward_ = ibp, forward

        results = convert_forward(
            model=model,
            input_tensors=input_tensors,
            layer_fn=layer_fn,
            input_dim=input_dim,
            dc_decomp=False,
            convex_domain=convex_domain,
            IBP=ibp_,
            forward=forward_,
            finetune=finetune_forward,
            shared=shared,
            softmax_to_linear=softmax_to_linear,
            forward_map=forward_map,
            back_bounds=back_bounds,
        )
        input_tensors, _, layer_map, forward_map = results

        # forward_map = switch_mode_mapping(forward, IBP=ibp, forward=forward, method=method)
    if get_direction(method) == BACKWARD_FEED.name:
        input_tensors, output, layer_map, forward_map = convert_backward(
            model=model,
            input_tensors=input_tensors,
            back_bounds=back_bounds,
            input_dim=input_dim,
            convex_domain=convex_domain,
            IBP=ibp,
            forward=forward,
            finetune=finetune_backward,
            layer_map=layer_map,
            forward_map=forward_map,
            final_ibp=final_ibp,
            final_forward=final_forward,
            **kwargs,
        )
    else:

        # check final_ibp and final_forward
        mode_from = get_mode(ibp, forward)
        mode_to = get_mode(final_ibp, final_forward)
        output = convert_2_mode(mode_from=mode_from, mode_to=mode_to, convex_domain=convex_domain)(results[1])

    # build decomon model
    return input_tensors, output, layer_map, forward_map


def clone(
    model,
    layer_fn=to_monotonic,
    dc_decomp=False,
    convex_domain={},
    ibp=True,
    forward=True,
    method="crown",
    back_bounds=[],
    finetune=False,
    shared=True,
    finetune_forward=False,
    finetune_backward=False,
    extra_inputs=[],
    to_keras=True,
    dico_grid={},
    **kwargs,
):

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    final_ibp, final_forward = ibp, forward

    ibp_, forward_ = get_ibp_forward_from_method(method)

    method = method.lower()
    assert method in [
        IBP.name,
        FORWARD.name,
        HYBRID.name,
        CROWN.name,
        CROWN_IBP.name,
        CROWN_FORWARD.name,
        CROWN_HYBRID.name,
    ]

    if not to_keras:
        # only available option: to_numpy (in the future do get and deserialize function
        return clone_to_numpy(
            model,
            ibp=ibp_,
            forward=forward_,
            method=method,
            final_ibp=final_ibp,
            final_forward=final_forward,
            convex_domain=convex_domain,
            shared=shared,
            back_bounds=back_bounds,
            dico_grid=dico_grid,
            **kwargs,
        )

    """
    if not ibp and not forward:
        # adapt the mode to the methods
        if len(convex_domain)==0 or convex_domain['name']!=Ball.name:
            if method in [CROWN_HYBRID.name, HYBRID.name]:
                ibp=True; forward=True
            if method in [IBP.name, CROWN_IBP.name, CROWN.name]:
                ibp=True
            if method in [FORWARD.name, CROWN_FORWARD.name]:
                forward=True
        else:
            # ball
            ibp=True; forward=True

    if not final_ibp and not final_forward:
        if method in [CROWN_IBP.name, CROWN.name]:
            final_ibp=ibp
            final_forward=True
        else:
            final_ibp=ibp
            final_forward=forward
    """

    if finetune:
        finetune_forward = True
        finetune_backward = True

    input_dim_init = -1
    input_dim = np.prod(model.input_shape[1:])

    """
    # temporary
    if method == IBP.name:
        ibp = True
        forward = False
    if method == FORWARD.name:
        ibp = False
        forward = True
    if method == CROWN_IBP.name:
        ibp = True
        forward = False
    if method == CROWN_FORWARD.name:
        ibp = False
        forward = True
    if method in [HYBRID.name, CROWN_HYBRID.name]:
        ibp = True
        forward = True
    """

    input_shape = None
    input_shape_vec = None

    for input_layer in model._input_layers:
        if len(input_layer.input_shape) > 1:
            raise ValueError("Expected one input tensor but got {}".format(len(input_layer.input_shape)))
        input_shape_vec_ = input_layer.input_shape[0]
        input_shape_ = tuple(list(input_shape_vec_)[1:])

        if input_shape_vec is None:
            input_shape_vec = input_shape_vec_
        if input_shape is None:
            input_shape = input_shape_
        else:
            if not np.allclose(input_shape, input_shape_):
                raise ValueError("Expected that every input layers use the same input_tensor")

    if isinstance(input_dim, tuple):
        input_dim_ = list(input_dim)[-1]
    else:
        input_dim_ = input_dim

    if len(convex_domain) == 0 or convex_domain["name"] != Ball.name:
        input_shape_x = (2, input_dim_)
    else:
        if isinstance(input_dim, tuple):
            input_shape_x = input_dim
        else:
            input_shape_x = (input_dim_,)

    z_tensor = Input(shape=input_shape_x, dtype=model.layers[0].dtype)

    if len(convex_domain) == 0 or convex_domain["name"] != Ball.name:

        u_c_tensor = Lambda(lambda z: z[:, 1])(z_tensor)
        l_c_tensor = Lambda(lambda z: z[:, 0])(z_tensor)

        z_value = K.cast(0.0, z_tensor.dtype)
        o_value = K.cast(1.0, z_tensor.dtype)
        W = Lambda(lambda z: tf.linalg.diag(z_value * z[:, 0] + o_value), dtype=z_tensor.dtype)(z_tensor)
        b = Lambda(lambda z: z_value * z[:, 1], dtype=z_tensor.dtype)(z_tensor)

    else:

        if convex_domain["p"] == np.inf:
            radius = convex_domain["eps"]

            u_c_tensor = Lambda(
                lambda var: var + K.cast(radius, dtype=model.layers[0].dtype), dtype=model.layers[0].dtype
            )(z_tensor)
            l_c_tensor = Lambda(
                lambda var: var - K.cast(radius, dtype=model.layers[0].dtype), dtype=model.layers[0].dtype
            )(z_tensor)

            z_value = K.cast(0.0, model.layers[0].dtype)
            o_value = K.cast(1.0, model.layers[0].dtype)

            W = tf.linalg.diag(z_value * u_c_tensor + o_value)
            b = z_value * u_c_tensor

        else:
            z_value = K.cast(0.0, model.layers[0].dtype)
            o_value = K.cast(1.0, model.layers[0].dtype)

            def get_bounds(z):
                W = tf.linalg.diag(z_value * z + o_value)
                b = z_value * z
                u_c_ = get_upper(z, W, b, convex_domain)
                l_c_ = get_lower(z, W, b, convex_domain)
                return [W, b, u_c_, l_c_]

            W, b, u_c_tensor, l_c_tensor = get_bounds(z_tensor)

    if ibp_ and forward_:
        # input_tensors = [z_tensor] + [u_c_tensor] * 3 + [l_c_tensor] * 3
        input_tensors = [z_tensor] + [u_c_tensor, W, b] + [l_c_tensor, W, b]
    if ibp_ and not forward_:
        input_tensors = [u_c_tensor, l_c_tensor]
    if not ibp_ and forward_:
        # input_tensors = [z_tensor] + [u_c_tensor] * 2 + [l_c_tensor] * 2
        input_tensors = [z_tensor] + [W, b] + [W, b]

    _, output, _, _ = convert(
        model,
        input_tensors=input_tensors,
        back_bounds=back_bounds,
        method=method,
        ibp=ibp_,
        forward=forward_,
        input_dim=-1,
        convex_domain=convex_domain,
        finetune=finetune,
        shared=shared,
        softmax_to_linear=True,
        layer_map={},
        forward_map={},
        finetune_forward=finetune_forward,
        finetune_backward=finetune_backward,
        final_ibp=final_ibp,
        final_forward=final_forward,
    )

    back_bounds_ = []
    for elem in back_bounds:
        if isinstance(elem._keras_history.layer, InputLayer):
            back_bounds_.append(elem)

    return DecomonModel(
        input=[z_tensor] + back_bounds_ + extra_inputs,
        output=output,
        convex_domain=convex_domain,
        dc_decomp=False,
        method=method,
        IBP=final_ibp,
        forward=final_forward,
        finetune=finetune,
        shared=shared,
        backward_bounds=(len(back_bounds) > 0),
    )
