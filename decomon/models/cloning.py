"""Module for MonotonicSequential.

It inherits from keras Sequential class.

"""
from __future__ import absolute_import

import inspect
import warnings
from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Input, InputLayer, Lambda, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list

from decomon.layers.utils import get_lower, get_upper, softmax_to_linear

from ..backward_layers.backward_layers import get_backward as get_backward_
from ..backward_layers.backward_layers import join
from ..backward_layers.utils import S_slope, V_slope, backward_linear_prod
from ..layers.core import Box, StaticVariables
from ..layers.decomon_layers import to_monotonic
from ..utils import M_BACKWARD, M_FORWARD, M_REC_BACKWARD
from .models import Backward, DecomonModel, DecomonSequential, Forward
from .utils import (
    check_input_tensors_functionnal,
    check_input_tensors_sequential,
    include_dim_layer_fn,
)

# from .backward_cloning import clone_backward_model


"""
def clone(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=-1,
    dc_decomp=False,
    convex_domain={},
    method="crown",
    slope_backward=V_slope.name,
    IBP=True,
    forward=True,
    finetune=False,
):
    ""
    :param model: Keras model
    :param input_tensors: List of input tensors to be used as inputs of our model or None
    :param layer_fn: cloning function to translate a layer into its decomon decomposition
    :param input_dim: the input dimension use to propagate our linear relaxation
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param mode: forward of backward
    :param finetune: ???
    :return: a decomon model
    ""

    if mode.lower() not in [M_FORWARD.name, M_BACKWARD.name, M_REC_BACKWARD.name]:
        raise ValueError()

    if not isinstance(model, Model):
        raise ValueError()

    if isinstance(model, Sequential):
        decomon_model = clone_sequential_model(
            model,
            input_tensors,
            layer_fn,
            input_dim=input_dim,
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            IBP=IBP,
            forward=forward,
            finetune=finetune,
            mode = mode
        )
    else:
        decomon_model = clone_functional_model(
            model,
            input_tensors,
            layer_fn,
            input_dim=input_dim,
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            IBP=IBP,
            forward=forward,
            finetune=finetune,
            mode = mode
        )

    return decomon_model


def clone_sequential_model(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=-1,
    dc_decomp=False,
    convex_domain={},
    IBP=True,
    forward=True,
    finetune=False,
    mode = M_BACKWARD.name
):
    ""Clone a `Sequential` model instance.
    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.

    :param model: Instance of `Sequential`.
    :param input_tensors: optional list of input tensors to build the model upon. If not provided, placeholder will
    be created.
    :param layer_fn: callable to be applied on non-input layers in the model. By default it clones the layer.
    Another example is to preserve the layer  to share the weights. This is required when we create a per-replica
    copy of the model with distribution strategy; we want the weights to be shared but still feed inputs
    separately so we create new input layers.
    :param input_dim: the input dimension use to propagate our linear relaxation
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param finetune: ???
    :return: An instance of `Sequential` reproducing the behavior of the original model with decomon layers.
    :raises: ValueError: in case of invalid `model` argument value or `layer_fn` argument value.
    ""
    if input_dim == -1:
        input_dim_init = -1
        input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_init = input_dim

    input_tensors = check_input_tensors_sequential(model, input_tensors, input_dim,\
                                   input_dim_init, IBP, forward, dc_decomp, convex_domain)

    model = softmax_to_linear(model)  # do better because you modify the model eventually

    layer_fn = include_dim_layer_fn(
        layer_fn,
        input_dim=input_dim,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=finetune,
    )  # return a list of Decomon layers

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

    # Use model._layers to ensure that all layers are cloned. The model's layers
    # property will exclude the initial InputLayer (if it exists) in the model,
    # resulting in a different Sequential model structure.
    def _get_layer(layer):
        if isinstance(layer, InputLayer):
            return []
        if isinstance(layer, Model):
            if isinstance(layer, Sequential):
                return [clone_sequential_model(
                    layer,
                    layer_fn=layer_fn,
                    input_dim=-1,
                    dc_decomp=dc_decomp,
                    convex_domain=convex_domain,
                    IBP=IBP,
                    forward=forward,
                    finetune=finetune,
                    mode=mode
                )]

            else:
                return [clone_functional_model(
                    layer,
                    layer_fn=layer_fn,
                    input_dim=-1,
                    dc_decomp=dc_decomp,
                    convex_domain=convex_domain,
                    IBP=IBP,
                    forward=forward,
                    finetune=finetune,
                    mode=mode,
                )]
        if isinstance(layer, Layer):
            return layer_fn(layer)
        raise ValueError

    forward_map = {}
    layer_map = {}
    output = input_tensors

    if mode in [M_FORWARD.name, M_BACKWARD.name]:


        monotonic_layers = []


        for layer in model.layers:

            forward_layers = _get_layer(layer)
            monotonic_layers += forward_layers
            layer_map[layer.name]=forward_layers
            for f_layer in forward_layers:
                output = f_layer(output)
                forward_map[f_layer.name] = output

        if mode == M_BACKWARD.name:
            output = clone_backward_model(model, input_tensors, layer_map, forward_map, IBP=IBP, \
                         forward=forward, finetune=finetune, convex_domain=convex_domain,
                         depth=0, back_bounds=[], \
                         upper_layer=None, lower_layer=None, output_tensors=output)



    else:
        # CROWN
        # retrieve first layer
        layer_init = model.layers[0]
        if isinstance(layer_init, Model):

            model_init_cloned = clone(layer_init,
                                input_tensors=input_tensors,
                                input_dim=input_dim,
                                      dc_decomp=dc_decomp,
                                      convex_domain=convex_domain,
                                      mode=mode,
                                      slope_backward=slope_backward,
                                      IBP=IBP,
                                      forward=forward,
                                      finetune=finetune
                                      )
            output_ = model_init_cloned(output)
            forward_map[layer_init.name] = output_

        else:
            forward_layers = _get_layer(layer_init)
            output_ = forward_layers[0](output)
            layer_map[layer_init.name]=forward_layers
            forward_map[forward_layers[0].name]=output_

        output = clone_backward_model(model, input_tensors, layer_map, forward_map, IBP=IBP, \
                         forward=forward, finetune=finetune, convex_domain=convex_domain,
                         depth=0, back_bounds=[], \
                         upper_layer=None, lower_layer=None, output_tensors=[])


    return DecomonModel(
        input_tensors,
        output,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=finetune,
        mode=mode
    )




def clone_functional_model(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=1,
    dc_decomp=False,
    convex_domain={},
    IBP=True,
    forward=True,
    mode='forward',
    finetune=False,
    mode = M_BACKWARD.name
):
    ""

    :param model:
    :param input_tensors: optional list of input tensors to build the model upon. If not provided, placeholder will
    be created.
    :param layer_fn: callable to be applied on non-input layers in the model. By default it clones the layer.
    :param input_dim: the input dimension use to propagate our linear relaxation
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param finetune: ???
    :return:
    ""

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)
    if isinstance(model, Sequential):
        raise ValueError(
            "Expected `model` argument " "to be a functional `Model` instance, " "got a `Sequential` instance instead:",
            model,
        )

    if input_dim == -1:
        input_dim_ = -1
        input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_ = input_dim

    layer_fn = include_dim_layer_fn(
        layer_fn,
        input_dim=input_dim,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=finetune,
    )
    model = softmax_to_linear(model)  # do not modify the previous model or send an alert message

    # we only handle one input
    #assert len(model._input_layers) == 1, "error: Expected one input only but got {}".format(len(model._input_layers))
    # if the model has several input layers, we assess whether they can use the same input tensor

    input_shape = None
    input_shape_vec = None


    for input_layer in model._input_layers:
        if len(input_layer.input_shape)>1:
            raise ValueError('Expected one input tensor but got {}'.format(len(input_layer.input_shape)))
        input_shape_vec_ = input_layer.input_shape[0]
        input_shape_ = tuple(list(input_shape_vec_)[1:])

        if input_shape_vec is None:
            input_shape_vec = input_shape_vec_
        if input_shape is None:
            input_shape = input_shape_
        else:
            if not np.allclose(input_shape, input_shape_):
                raise ValueError('Expected that every input layers use the same input_tensor')

    layer = model._input_layers[0]
    if input_tensors is None:

        if isinstance(input_dim, tuple):
            input_dim_ = list(input_dim)[-1]
        else:
            input_dim_ = input_dim

        if len(convex_domain) == 0:
            input_shape_x = (2, input_dim_)
        else:
            if isinstance(input_dim, tuple):
                input_shape_x = input_dim
            else:
                input_shape_x = (input_dim_,)

        input_shape_w = tuple([input_dim_] + list(input_shape_vec)[1:])

        z_tensor = Input(shape=input_shape_x, dtype=layer.dtype, name="z_" + layer.name)
        y_tensor = Input(shape=input_shape, dtype=layer.dtype, name="y_" + layer.name)

        if forward:
            b_u_tensor = Input(shape=input_shape, dtype=layer.dtype, name="b_u_" + layer.name)
            b_l_tensor = Input(shape=input_shape, dtype=layer.dtype, name="b_l_" + layer.name)
        if IBP:
            u_c_tensor = Input(shape=input_shape, dtype=layer.dtype, name="u_c_" + layer.name)
            l_c_tensor = Input(shape=input_shape, dtype=layer.dtype, name="l_c_" + layer.name)

        if forward:
            if input_dim_ > 0:
                w_u_tensor = Input(shape=input_shape_w, dtype=layer.dtype, name="w_u_" + layer.name)
                w_l_tensor = Input(shape=input_shape_w, dtype=layer.dtype, name="w_l_" + layer.name)
            else:
                w_u_tensor = Input(shape=input_shape, dtype=layer.dtype, name="w_u_" + layer.name)
                w_l_tensor = Input(shape=input_shape, dtype=layer.dtype, name="w_l_" + layer.name)

        if IBP:
            if forward:  # hybrid
                input_tensors = [
                    y_tensor,
                    z_tensor,
                    u_c_tensor,
                    w_u_tensor,
                    b_u_tensor,
                    l_c_tensor,
                    w_l_tensor,
                    b_l_tensor,
                ]
            else:
                # IBP only
                input_tensors = [
                    y_tensor,
                    z_tensor,
                    u_c_tensor,
                    l_c_tensor,
                ]
        else:
            # forward only
            input_tensors = [
                y_tensor,
                z_tensor,
                w_u_tensor,
                b_u_tensor,
                w_l_tensor,
                b_l_tensor,
            ]

        if dc_decomp:
            h_tensor = Input(shape=input_shape, dtype=layer.dtype, name="h_" + layer.name)
            g_tensor = Input(shape=input_shape, dtype=layer.dtype, name="g_" + layer.name)
            input_tensors += [h_tensor, g_tensor]

        # Cache newly created input layer.
        input_tensors = [input_tensor._keras_history[0] for input_tensor in input_tensors]
    else:
        input_tensors = to_list(input_tensors)
        _input_tensors = []
        if IBP:
            if forward:
                names_i = ["y", "z", "u_c", "w_u", "b_u", "l_c", "w_l", "b_l"]
            else:
                names_i = ["y", "z", "u_c", "l_c"]
        else:
            names_i = ["y", "z", "w_u", "b_u", "w_l", "b_l"]

        if dc_decomp:
            names_i += ["h", "g"]
            if IBP and forward:
                assert len(input_tensors) == 10, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if IBP and not forward:
                assert len(input_tensors) == 6, "wrong number of inputs, expexted 6 but got {}".format(
                    len(input_tensors)
                )
            if not IBP and forward:
                assert len(input_tensors) == 8, "wrong number of inputs, expexted 8 but got {}".format(
                    len(input_tensors)
                )
        else:
            if IBP and forward:
                assert len(input_tensors) == 8, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if IBP and not forward:
                assert len(input_tensors) == 4, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if not IBP and forward:
                assert len(input_tensors) == 6, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )

        for i, x in enumerate(input_tensors):

            if not K.is_keras_tensor(x):
                name = layer.name
                input_tensor = Input(tensor=x, name=name + "_{}".format(names_i[i]))
                _input_tensors.append(input_tensor)
            else:
                _input_tensors.append(x)
        input_tensors = _input_tensors

    #
    forward_map_tensor={}
    if mode !=Forward.name:
        forward_map_decomon={}

    for layer_input in model._input_layers:
        # convert layer into a Decomon layer or a Decomon list of layers
        if isinstance(layer_input, InputLayer):
            forward_map_tensor[layer_input.name] = input_tensors
        else:
            layer_input_decomon = layer_fn(layer_input)
            output = input_tensors
            for layer_i in layer_input_decomon:
                output = layer_i(output)
            forward_map_tensor[layer_input.name] = output


    # sort by depth
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            layer_ = node.outbound_layer

            if not layer_.name in forward_map_tensor:
                # retrieve input layers
                input_ = to_list(node.inbound_layers)
                output = []
                for input_i in input_:
                    output+= forward_map_tensor[input_i.name]

                layer_decomon = layer_fn(layer_)
                for layer_decomon_i in layer_decomon:
                    output = layer_decomon_i(output)
                forward_map_tensor[layer_.name] = output

    # retrieve the output
    depth_output = 0
    nodes_output = model._nodes_by_depth[depth_output]
    output_tensors=[]
    for node in nodes:
        layer_output = node.outbound_layer

        output_tensors+= forward_map_tensor[layer_output.name]

    if mode==Forward.name:
        decomon_model = DecomonModel(
            input_tensors,
            output_tensors,
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            mode=mode,
            IBP=IBP,
            forward=forward,
        )

    else:
        # do it for all outputs
        depth_init = 0  # to set # parameters
        layer_map_backward_init={}

        layer_map_backward={} # can be init
        layer_map_backward.update(layer_map_backward_init)
        # consider only crown-?
        depth_keys = list(model._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        for depth in range(depth_init, np.max(depth_keys)):

            nodes_depth = model._nodes_by_depth[depth]

            output_layers = [node.outbound_layer for node in nodes_depth]

            for node in nodes_depth:
                layer_ = node.outbound_layer
                if isinstance(layer_, InputLayer):
                    raise NotImplementedError() # a faire

            # convert to backward
            layer_back = get_backward_(layer_) # attention extra parameters...
            # retrieve input layers of layer_
            input_ = to_list(node.inbound_layers)
            if np.min([input_i.name in forward_map_tensor.keys()]):
                # forward propagation
                input_tensors_back=[]
                for input_i in input_:
                    input_tensors_back+=forward_map_tensor[input_i.name]
                if layer_.name in layer_map_backward:
                    input_tensors_back+=layer_map_backward[layer_.name]
                output_backward = layer_back(input_tensors_back)
                for back_bounds_i, input_i in zip(output_backward, input_):
                    if input_i.name not in layer_map_backward:
                        layer_map_backward[input_i] = back_bounds_i
            else:
                raise NotImplementedError() # CROWN

            # assume that they are in layer_map (forward-backward approach)

"""
