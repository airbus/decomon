"""Module for MonotonicSequential.

It inherits from keras Sequential class.

"""


import numpy as np
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.generic_utils import to_list

from decomon.layers.decomon_layers import to_monotonic
from decomon.layers.utils import softmax_to_linear as softmax_2_linear
from decomon.models.utils import (
    check_input_tensors_sequential,
    get_depth_dict,
    get_inner_layers,
    include_dim_layer_fn,
)


def convert_forward(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=-1,
    dc_decomp=False,
    convex_domain=None,
    IBP=True,
    forward=True,
    finetune=False,
    shared=True,
    softmax_to_linear=True,
    back_bounds=None,
    joint=True,
    **kwargs,
):

    if convex_domain is None:
        convex_domain = {}
    if back_bounds is None:
        back_bounds = []
    if not isinstance(model, Model):
        raise ValueError()

    f_output = convert_forward_functional_model(
        model,
        input_tensors,
        layer_fn,
        input_dim=input_dim,
        convex_domain=convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=finetune,
        shared=shared,
        softmax_to_linear=True,
    )

    return f_output


def convert_forward_functional_model(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=1,
    convex_domain=None,
    IBP=True,
    forward=True,
    finetune=False,
    shared=True,
    softmax_to_linear=True,
    count=0,
    joint=True,
    **kwargs,
):

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    if input_dim == -1:
        input_dim_init = -1
        if isinstance(model.input_shape, list):
            input_dim = np.prod(model.input_shape[0][1:])
        else:
            input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_init = input_dim

    if input_tensors is None:
        # check that the model has one input else
        input_tensors = []
        for i in range(len(model._input_layers)):

            tmp = check_input_tensors_sequential(model, None, input_dim, input_dim, IBP, forward, False, convex_domain)
            input_tensors += tmp

    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)

    has_iter = False
    if layer_fn is not None and len(layer_fn.__code__.co_varnames) == 1 and "layer" in layer_fn.__code__.co_varnames:
        has_iter = True

    if not has_iter:
        layer_fn_ = include_dim_layer_fn(
            layer_fn,
            input_dim=input_dim,
            convex_domain=convex_domain,
            IBP=IBP,
            forward=forward,
            finetune=finetune,
            shared=shared,
        )  # return a list of Decomon layers

        def func(layer):
            return layer_fn_(layer)

        layer_fn = func

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

    # create input tensors

    # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    output_map = {}
    layer_map = {}
    output = input_tensors
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:

            layer = node.outbound_layer
            parents = node.parent_nodes
            if id(node) in output_map.keys() and joint:
                continue
            if len(parents):
                output = []
                for parent in parents:
                    output += output_map[id(parent)]

            if isinstance(layer, Model):
                _, output, layer_map_, output_map_ = convert_forward_functional_model(
                    model=layer,
                    input_tensors=output,
                    layer_fn=layer_fn,
                    input_dim=input_dim_init,
                    convex_domain=convex_domain,
                    IBP=IBP,
                    forward=forward,
                    finetune=finetune,
                    shared=shared,
                    softmax_to_linear=softmax_to_linear,
                    count=count,
                    **kwargs,
                )
                count = count + get_inner_layers(layer)
                layer_map[id(node)] = layer_map_
                output_map[id(node)] = output_map_
            else:

                list_layer_decomon = layer_fn(layer)
                layer_map[id(node)] = []
                for layer_decomon in list_layer_decomon:
                    layer_decomon._name = f"{layer_decomon.name}_{count}"
                    count += 1
                    output = layer_decomon(output)
                    layer_map[id(node)].append(layer_decomon)
                    if len(list_layer_decomon) > 1:
                        output_map[f"{id(node)}_{layer_decomon.name}"] = output

            output_map[id(node)] = output

    output = []
    output_nodes = dico_nodes[0]
    # the ordering may change
    output_names = [tensor._keras_history.layer.name for tensor in to_list(model.output)]
    for output_name in output_names:
        for node in output_nodes:
            if node.outbound_layer.name == output_name:
                output += output_map[id(node)]

    return input_tensors, output, layer_map, output_map
