from __future__ import absolute_import

import numpy as np
from tensorflow.keras.layers import Input, InputLayer, Lambda, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list

from decomon.layers.decomon_layers import DecomonMaximum, DecomonMinimum

from ..backward_layers.backward_layers import get_backward as get_backward_
from ..utils import (
    F_FORWARD,
    F_HYBRID,
    F_IBP,
    backward_maximum,
    backward_minimum,
    get_lower_layer,
    get_upper_layer,
)
from .utils import (
    check_input_tensors_sequential,
    convert_to_backward_bounds,
    get_back_bounds_model,
    get_FORWARD,
    get_IBP,
    get_input_dim,
    get_key,
    get_mode,
    get_node_by_id,
)


def update_input(back_bounds, x_tensor, mode, upper_layer=None, lower_layer=None):

    x = x_tensor
    w_u, b_u, w_l, b_l = back_bounds
    if mode == F_FORWARD.name:
        return [x, w_u, b_u, w_l, b_l]

    u_c = upper_layer([x, w_u, b_u])
    l_c = lower_layer([x, w_l, b_l])

    if mode == F_HYBRID.name:
        return [x, u_c, w_u, b_u, l_c, w_l, b_l]

    if mode == F_IBP.name:
        return [u_c, l_c]


def get_backward_layer(
    layer,
    back_bounds,
    mode,
    finetune=False,
    convex_domain=None,
    layer_map=None,
    forward_map=None,
    shared=True,
    softmax_to_linear=True,
):

    if convex_domain is None:
        convex_domain = {}
    if layer_map is None:
        layer_map = {}
    if forward_map is None:
        forward_map = {}
    if isinstance(layer, Model):
        # call clone
        if layer.name in layer_map:
            input_tensor, output_tensor, _ = convert_backward(
                layer,
                None,
                back_bounds,
                convex_domain=convex_domain,
                IBP=get_IBP(mode),
                forward=get_FORWARD(mode),
                finetune=finetune,
                shared=shared,
                softmax_to_linear=softmax_to_linear,
                kwargs={"layer_map": layer_map[layer.name], "forward_map": forward_map},
            )

        else:
            input_tensor, output_tensor, layer_map_model = convert_backward(
                layer,
                None,
                back_bounds,
                convex_domain=convex_domain,
                IBP=get_IBP(mode),
                forward=get_FORWARD(mode),
                finetune=finetune,
                shared=shared,
                softmax_to_linear=softmax_to_linear,
            )
            layer_map[layer.name] = layer_map_model
        model_backward = Model(input_tensor, output_tensor)
        return model_backward

    previous = True
    if back_bounds == 0:
        previous = False

    try:
        return get_backward_(layer, previous=previous, mode=mode, convex_domain=convex_domain)
    except KeyError:
        import pdb

        pdb.set_trace()


def clone_backward_layer(
    node,
    layer_map,
    forward_map,
    mode,
    back_bounds,
    finetune=False,
    convex_domain=None,
    input_tensors=None,
    upper_layer=None,
    lower_layer=None,
    x_tensor=None,
):

    if convex_domain is None:
        convex_domain = {}
    layer_ = node.outbound_layer
    id_node = get_node_by_id(node)

    if isinstance(layer_, InputLayer):

        if len(back_bounds) == 0:
            return input_tensors, layer_map, forward_map
        else:
            return update_input(back_bounds, x_tensor, mode, upper_layer, lower_layer), layer_map, forward_map

    if f"{layer_.name}_{id_node}" not in layer_map:
        # crown mode

        layer_back = get_backward_layer(
            layer_, back_bounds=len(back_bounds), mode=mode, finetune=finetune, convex_domain=convex_domain
        )
        input_nodes = node.parent_nodes

        if len(input_nodes):
            input_tensors_crown = []
            for node_i in input_nodes:
                # import pdb;
                # pdb.set_trace()  # this should not happen now
                t_i = clone_backward_layer(
                    node_i,
                    layer_map,
                    forward_map,
                    mode,
                    [],
                    finetune,
                    convex_domain=convex_domain,
                    input_tensors=input_tensors,
                    upper_layer=upper_layer,
                    lower_layer=lower_layer,
                    x_tensor=x_tensor,
                )
                input_tensors_crown += t_i[0]
        back_bounds = layer_back(input_tensors_crown + back_bounds)
        if isinstance(back_bounds, tuple):
            back_bounds = list(back_bounds)

    else:

        if isinstance(layer_, Model):

            # get inputs
            inbound_nodes = to_list(node.parent_nodes)
            if len(inbound_nodes):
                # tensorflow can create fake inputs when we reach the first layers of the model, those are not stored
                # in our forward  map
                input_tensor_list = []
                for inbound_node in inbound_nodes:
                    input_layer_i = inbound_node.outbound_layer
                    node_id_i = get_node_by_id(inbound_node)
                    if f"{input_layer_i.name}_{node_id_i}" in forward_map:
                        input_tensor_list.append(forward_map[f"{input_layer_i.name}_{node_id_i}"])
                    else:
                        if input_layer_i.name[-6::] == "_input":
                            input_tensor_list.append(input_tensors)
                        else:
                            import pdb

                            pdb.set_trace()
                            raise KeyError()

                input_tensors_layer = []

                for t_i in input_tensor_list:
                    input_tensors_layer += t_i

            back_bounds_model = get_back_bounds_model(back_bounds, layer_)
            results = convert_backward(
                layer_,
                input_tensors=None,
                back_bounds=back_bounds_model,
                convex_domain=convex_domain,
                IBP=get_IBP(mode),
                forward=get_FORWARD(mode),
                finetune=finetune,
                layer_map=layer_map[f"{layer_.name}_{id_node}"],
                forward_map=forward_map,
                x_tensor=x_tensor,
            )

            import pdb

            pdb.set_trace()
            tmp_model = Model(results[0], results[1])

            import pdb

            pdb.set_trace()
            back_bounds_ = convert_to_backward_bounds(
                mode, tmp_model(input_tensors_layer + back_bounds), layer_.get_input_shape_at(0)[-1]
            )
            if isinstance(back_bounds_, tuple):
                back_bounds_ = list(back_bounds_)
            back_bounds = back_bounds_

        else:

            # layer_list = to_list(layer_map[layer_.name])
            # update the key
            layer_list = to_list(layer_map[f"{layer_.name}_{id_node}"])
            layer_list = layer_list[::-1]

            for i in range(len(layer_list) - 1):

                layer_back = get_backward_layer(
                    layer_list[i],
                    back_bounds=len(back_bounds),
                    mode=mode,
                    finetune=finetune,
                    convex_domain=convex_domain,
                )
                if f"{layer_list[i + 1].name}_{id_node}" not in forward_map:
                    input_tensor_i = forward_map[get_key(layer_list[i + 1], id_node, forward_map)]
                else:
                    input_tensor_i = forward_map[f"{layer_list[i + 1].name}_{id_node}"]
                back_bounds_ = layer_back(input_tensor_i + back_bounds)
                if back_bounds_[0].shape[-1] != back_bounds_[1].shape[-1]:
                    import pdb

                    pdb.set_trace()
                back_bounds = back_bounds_
                if isinstance(back_bounds, tuple):
                    back_bounds = list(back_bounds)

            layer_back = get_backward_layer(
                layer_list[-1], back_bounds=len(back_bounds), mode=mode, finetune=finetune, convex_domain=convex_domain
            )
            # input_layers = to_list(node.inbound_layers)
            inbound_nodes = to_list(node.parent_nodes)
            if len(inbound_nodes):
                # tensorflow can create fake inputs when we reach the first layers of the model, those are not stored
                # in our forward  map
                input_tensor_list = []
                for inbound_node in inbound_nodes:
                    input_layer_i = inbound_node.outbound_layer
                    node_id_i = get_node_by_id(inbound_node)
                    if f"{input_layer_i.name}_{node_id_i}" in forward_map:
                        input_tensor_list.append(forward_map[f"{input_layer_i.name}_{node_id_i}"])
                    else:
                        if input_layer_i.name[-6::] == "_input":
                            input_tensor_list.append(input_tensors)
                        else:
                            import pdb

                            pdb.set_trace()
                            raise KeyError()

                input_tensors_layer = []

                for t_i in input_tensor_list:
                    input_tensors_layer += t_i

            back_bounds_ = layer_back(input_tensors_layer + back_bounds)

            if back_bounds_[0].shape[-1] != back_bounds_[1].shape[-1]:
                import pdb

                pdb.set_trace()
            back_bounds = back_bounds_
            if isinstance(back_bounds, tuple):
                back_bounds = list(back_bounds)

    inbound_nodes = node.parent_nodes
    if len(inbound_nodes) == 0:
        return update_input(back_bounds, x_tensor, mode, upper_layer, lower_layer)
    else:
        n = int(len(back_bounds) / len(inbound_nodes))
        back_bounds_inbound = [back_bounds[i * n : (i + 1) * n] for i in range(len(inbound_nodes))]
        """
        back_bounds_rec = [ update_input(clone_backward_layer(inbound_nodes[i], layer_map, forward_map, \
                                         mode, back_bounds_inbound[i], finetune, input_tensors=input_tensors, upper_layer=upper_layer, lower_layer=lower_layer),
                                         input_tensors, mode, upper_layer=upper_layer, lower_layer=lower_layer)
                                         for i in range(len(inbound_nodes))]
        """
        back_bounds_rec = []
        for i in range(len(inbound_nodes)):
            bounds_rec, layer_map, forward_map = clone_backward_layer(
                inbound_nodes[i],
                layer_map,
                forward_map,
                mode,
                back_bounds_inbound[i],
                finetune,
                input_tensors=input_tensors,
                upper_layer=upper_layer,
                lower_layer=lower_layer,
                x_tensor=x_tensor,
            )
            back_bounds_rec += bounds_rec

        # update
        max_bounds = DecomonMaximum(mode=mode, convex_domain=convex_domain)(back_bounds_rec)
        min_bounds = DecomonMinimum(mode=mode, convex_domain=convex_domain)(back_bounds_rec)

        def func(inputs_):
            n = int(len(inputs_) / 2)
            inputs_0 = inputs_[:n]
            inputs_1 = inputs_[n:]
            if mode == F_IBP.name:
                u_c = inputs_0[0]
                l_c = inputs_1[1]
                return [u_c, l_c]
            if mode == F_FORWARD.name:
                x, w_u, b_u = inputs_0[:3]
                w_l, b_l = inputs_1[-2:]
                return [x, w_u, b_u, w_l, b_l]
            if mode == F_HYBRID.name:
                x, u_c, w_u, b_u = inputs_0[:4]
                l_c, w_l, b_l = inputs_1[-3:]
                return [x, u_c, w_u, b_u, l_c, w_l, b_l]

        lambda_f = Lambda(func)

        output = lambda_f(max_bounds + min_bounds)
        return output, layer_map, forward_map


def convert_backward(
    model,
    input_tensors=None,
    back_bounds=None,
    input_dim=-1,
    convex_domain=None,
    IBP=True,
    forward=True,
    finetune=False,
    shared=True,
    softmax_to_linear=True,
    layer_map=None,
    forward_map=None,
    **kwargs,
):
    if convex_domain is None:
        convex_domain = {}
    if layer_map is None:
        layer_map = {}
    if forward_map is None:
        forward_map = {}
    if not isinstance(model, Model):
        raise ValueError()

    if input_dim == -1:
        input_dim_init = -1
        input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_init = input_dim

    if input_tensors is None:
        input_tensors = check_input_tensors_sequential(
            model,
            input_tensors,
            input_dim,
            input_dim_init,
            IBP,
            forward,
            dc_decomp=False,
            convex_domain=convex_domain,
        )

    if "upper_layer" in kwargs:
        upper_layer = kwargs["upper_layer"]
        lower_layer = kwargs["lower_layer"]
    else:
        upper_layer = get_upper_layer(convex_domain)
        lower_layer = get_lower_layer(convex_domain)

    mode = get_mode(IBP=IBP, forward=forward)

    depth = 0  # start from the beginning
    nodes = model._nodes_by_depth[depth]

    if back_bounds is None or len(back_bounds) == 0:
        back_bounds = [[] * len(nodes)]

    outputs = []
    for node_i, bounds_i in zip(nodes, back_bounds):
        if mode == "ibp":
            x_tensor = kwargs["x_tensor"]
        else:
            x_tensor = input_tensors[0]
        output_i, layer_map, forward_map = clone_backward_layer(
            node_i,
            layer_map=layer_map,
            forward_map=forward_map,
            mode=mode,
            back_bounds=bounds_i,
            finetune=finetune,
            convex_domain=convex_domain,
            input_tensors=input_tensors,
            upper_layer=upper_layer,
            lower_layer=lower_layer,
            x_tensor=x_tensor,
        )
        outputs += output_i

    # add back_bounds from input_tensors
    for bounds_i in back_bounds:
        if isinstance(bounds_i, list):
            for bound_i_j in bounds_i:
                input_tensors += bound_i_j
        else:
            input_tensors += bounds_i

    return input_tensors, outputs, layer_map, forward_map


def clone_backward_layer_last(
    node,
    layer_map,
    forward_map,
    mode,
    back_bounds,
    finetune=False,
    convex_domain=None,
    input_tensors=None,
    upper_layer=None,
    lower_layer=None,
    x_tensor=None,
    combine_with_input=False,
):

    if convex_domain is None:
        convex_domain = {}
    layer_ = node.outbound_layer
    id_node = get_node_by_id(node)
    if isinstance(layer_, InputLayer):

        layer_back = get_backward_layer(
            layer_, back_bounds=len(back_bounds), mode=mode, finetune=finetune, convex_domain=convex_domain
        )
        back_bounds_ = layer_back(input_tensors + back_bounds)

        if isinstance(back_bounds_, tuple):
            back_bounds_ = list(back_bounds_)

        back_bounds = back_bounds_

        if combine_with_input:
            result = fuse_forward_backward(
                mode,
                input_tensors,
                back_bounds,
                upper_layer=upper_layer,
                lower_layer=lower_layer,
                x_tensor=x_tensor,
                convex_domain=convex_domain,
            )
        else:
            result = update_input(back_bounds, x_tensor, mode, upper_layer, lower_layer)

        return result, layer_map, forward_map

        # if len(back_bounds)==0:
        #    return input_tensors, layer_map, forward_map
        # else:
        #    return update_input(back_bounds, x_tensor, mode, upper_layer, lower_layer), layer_map, forward_map

    if f"{layer_.name}_{id_node}" not in layer_map:
        # crown mode

        input_nodes = node.parent_nodes

        if len(input_nodes):
            input_tensors_crown = []
            for node_i in input_nodes:
                t_i = clone_backward_layer(
                    node_i,
                    layer_map,
                    forward_map,
                    mode,
                    [],
                    finetune,
                    convex_domain=convex_domain,
                    input_tensors=input_tensors,
                    upper_layer=upper_layer,
                    lower_layer=lower_layer,
                    x_tensor=x_tensor,
                    combine_with_input=combine_with_input,
                )
                input_tensors_crown += to_list(t_i[0])

        if mode == F_IBP.name:
            input_dim = layer_.get_input_shape_at(0)[-1]
        else:
            input_dim = input_tensors[0].shape[-1]
        layer_back = get_backward_layer(
            layer_,
            back_bounds=len(back_bounds),
            mode=mode,
            finetune=finetune,
            convex_domain=convex_domain,
            input_tensors=input_tensors_crown,
            input_dim=input_dim,
        )
        back_bounds = layer_back(input_tensors_crown + back_bounds)
        if isinstance(back_bounds, tuple):
            back_bounds = list(back_bounds)

    else:

        if isinstance(layer_, Model):

            # get inputs
            inbound_nodes = to_list(node.parent_nodes)
            if len(inbound_nodes):
                # tensorflow can create fake inputs when we reach the first layers of the model, those are not stored
                # in our forward  map
                input_tensor_list = []
                f_map = forward_map[f"{layer_.name}_{id_node}"][1]
                for inbound_node in inbound_nodes:
                    input_layer_i = inbound_node.outbound_layer
                    node_id_i = get_node_by_id(inbound_node)
                    if f"{input_layer_i.name}_{node_id_i}" in f_map:
                        if isinstance(input_layer_i, Model):
                            input_tensor_list.append(f_map[f"{input_layer_i.name}_{node_id_i}"][0])
                        else:
                            input_tensor_list.append(f_map[f"{input_layer_i.name}_{node_id_i}"])
                    else:
                        import pdb

                        pdb.set_trace()
                        if input_layer_i.name[-6::] == "_input":
                            input_tensor_list.append(input_tensors)
                        else:
                            import pdb

                            pdb.set_trace()
                            raise KeyError()

                input_tensors_layer = []

                for t_i in input_tensor_list:
                    input_tensors_layer += t_i

            back_bounds_model = get_back_bounds_model(back_bounds, layer_)

            if mode == F_IBP.name:
                _, output_tensors_layer, _, _ = convert_backward(
                    layer_,
                    input_tensors=input_tensors_layer,
                    back_bounds=back_bounds_model,
                    convex_domain=convex_domain,
                    IBP=get_IBP(mode),
                    forward=get_FORWARD(mode),
                    finetune=finetune,
                    layer_map=layer_map[f"{layer_.name}_{id_node}"],
                    forward_map=forward_map,
                    fuse_with_input=False,
                    x_tensor=x_tensor,
                )
            else:
                _, output_tensors_layer, _, _ = convert_backward(
                    layer_,
                    input_tensors=input_tensors_layer,
                    back_bounds=back_bounds_model,
                    convex_domain=convex_domain,
                    IBP=get_IBP(mode),
                    forward=get_FORWARD(mode),
                    finetune=finetune,
                    layer_map=layer_map[f"{layer_.name}_{id_node}"],
                    forward_map=forward_map,
                    fuse_with_input=False,
                )

            back_bounds_ = convert_to_backward_bounds(
                mode, output_tensors_layer, np.prod(layer_.get_input_shape_at(0)[1:])
            )

            if isinstance(back_bounds_, tuple):
                back_bounds_ = list(back_bounds_)
            back_bounds = back_bounds_

        else:

            # layer_list = to_list(layer_map[layer_.name])
            # update the key
            layer_list = to_list(layer_map[f"{layer_.name}_{id_node}"])
            layer_list = layer_list[::-1]

            for i in range(len(layer_list) - 1):
                layer_back = get_backward_layer(
                    layer_list[i],
                    back_bounds=len(back_bounds),
                    mode=mode,
                    finetune=finetune,
                    convex_domain=convex_domain,
                )
                if f"{layer_list[i + 1].name}_{id_node}" not in forward_map:
                    input_tensor_i = forward_map[get_key(layer_list[i + 1], id_node, forward_map)]
                else:
                    input_tensor_i = forward_map[f"{layer_list[i + 1].name}_{id_node}"]
                back_bounds_ = layer_back(input_tensor_i + back_bounds)
                if back_bounds_[0].shape[-1] != back_bounds_[1].shape[-1]:
                    import pdb

                    pdb.set_trace()
                back_bounds = back_bounds_
                if isinstance(back_bounds, tuple):
                    back_bounds = list(back_bounds)

            layer_back = get_backward_layer(
                layer_list[-1], back_bounds=len(back_bounds), mode=mode, finetune=finetune, convex_domain=convex_domain
            )
            # input_layers = to_list(node.inbound_layers)
            inbound_nodes = to_list(node.parent_nodes)
            if len(inbound_nodes):
                # tensorflow can create fake inputs when we reach the first layers of the model, those are not stored
                # in our forward  map
                input_tensor_list = []
                for inbound_node in inbound_nodes:
                    input_layer_i = inbound_node.outbound_layer
                    node_id_i = get_node_by_id(inbound_node)

                    if isinstance(input_layer_i, Model):
                        toto = forward_map[f"{input_layer_i.name}_{node_id_i}"][0]
                        input_tensor_list.append(toto)
                    else:

                        if f"{input_layer_i.name}_{node_id_i}" in forward_map:
                            input_tensor_list.append(forward_map[f"{input_layer_i.name}_{node_id_i}"])
                        else:
                            if input_layer_i.name[-6::] == "_input":
                                input_tensor_list.append(input_tensors)
                            else:
                                import pdb

                                pdb.set_trace()
                                raise KeyError()

                input_tensors_layer = []

                for t_i in input_tensor_list:
                    input_tensors_layer += t_i
            back_bounds_ = layer_back(input_tensors_layer + back_bounds)
            back_bounds = back_bounds_
            if isinstance(back_bounds, tuple):
                back_bounds = list(back_bounds)

    inbound_nodes = node.parent_nodes
    if len(inbound_nodes) == 0:
        # FAUX !!!!!
        return update_input(back_bounds, x_tensor, mode, upper_layer, lower_layer)
    else:
        n = len(inbound_nodes)
        if n == 1:
            back_bounds_inbound = [back_bounds]
        else:
            back_bounds_inbound = back_bounds  # to be improved
            if len(back_bounds) != n:
                import pdb

                pdb.set_trace()

        # back_bounds_inbound = [back_bounds[i*n:(i+1)*n] for i in range(len(inbound_nodes))]
        back_bounds_rec = []
        for i in range(len(inbound_nodes)):
            bounds_rec, layer_map, forward_map = clone_backward_layer(
                inbound_nodes[i],
                layer_map,
                forward_map,
                mode,
                back_bounds_inbound[i],
                finetune,
                input_tensors=input_tensors,
                upper_layer=upper_layer,
                lower_layer=lower_layer,
                x_tensor=x_tensor,
                combine_with_input=combine_with_input,
            )
            back_bounds_rec += bounds_rec

        # update
        max_bounds = DecomonMaximum(mode=mode, convex_domain=convex_domain)(back_bounds_rec)
        min_bounds = DecomonMinimum(mode=mode, convex_domain=convex_domain)(back_bounds_rec)

        def func(inputs_):
            n = int(len(inputs_) / 2)
            inputs_0 = inputs_[:n]
            inputs_1 = inputs_[n:]
            if mode == F_IBP.name:
                u_c = inputs_0[0]
                l_c = inputs_1[1]
                return [u_c, l_c]
            if mode == F_FORWARD.name:
                x, w_u, b_u = inputs_0[:3]
                w_l, b_l = inputs_1[-2:]
                return [x, w_u, b_u, w_l, b_l]
            if mode == F_HYBRID.name:
                x, u_c, w_u, b_u = inputs_0[:4]
                l_c, w_l, b_l = inputs_1[-3:]
                return [x, u_c, w_u, b_u, l_c, w_l, b_l]

        lambda_f = Lambda(func)

        output = lambda_f(max_bounds + min_bounds)
        return output, layer_map, forward_map
