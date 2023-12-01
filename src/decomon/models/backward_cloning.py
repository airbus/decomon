from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
from keras.config import floatx
from keras.layers import Concatenate, Lambda, Layer
from keras.models import Model
from keras.src.ops.node import Node
from keras.src.utils.python_utils import to_list

from decomon.backward_layers.backward_merge import BackwardMerge
from decomon.backward_layers.convert import to_backward
from decomon.backward_layers.core import BackwardLayer
from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    Slope,
    get_affine,
    get_mode,
)
from decomon.keras_utils import BatchedIdentityLike
from decomon.layers.utils import softmax_to_linear as softmax_2_linear
from decomon.models.crown import Convert2BackwardMode, Fuse, MergeWithPrevious
from decomon.models.forward_cloning import OutputMapDict
from decomon.models.utils import Convert2Mode, ensure_functional_model, get_depth_dict
from decomon.types import BackendTensor, Tensor


def get_disconnected_input(
    mode: Union[str, ForwardMode],
    perturbation_domain: PerturbationDomain,
    dtype: Optional[str] = None,
) -> Layer:
    mode = ForwardMode(mode)
    dc_decomp = False
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    affine = get_affine(mode)
    if dtype is None:
        dtype = floatx()

    def disco_priv(inputs: List[Tensor]) -> List[Tensor]:
        x, u_c, w_f_u, b_f_u, l_c, w_f_l, b_f_l, h, g = inputs_outputs_spec.get_fullinputs_from_inputsformode(inputs)
        dtype = x.dtype
        empty_tensor = inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        if affine:
            x = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
            w_u = BatchedIdentityLike()(u_c)
            b_u = K.zeros_like(u_c)
        else:
            w_u, b_u = empty_tensor, empty_tensor

        return inputs_outputs_spec.extract_outputsformode_from_fulloutputs([x, u_c, w_u, b_u, l_c, w_u, b_u])

    return Lambda(disco_priv, dtype=dtype)


def retrieve_layer(
    node: Node,
    layer_fn: Callable[[Layer], BackwardLayer],
    backward_map: Dict[int, BackwardLayer],
    joint: bool = True,
) -> BackwardLayer:
    if id(node) in backward_map:
        backward_layer = backward_map[id(node)]
    else:
        backward_layer = layer_fn(node.operation)
        if joint:
            backward_map[id(node)] = backward_layer
    return backward_layer


def crown_(
    node: Node,
    ibp: bool,
    affine: bool,
    perturbation_domain: PerturbationDomain,
    input_map: Dict[int, List[keras.KerasTensor]],
    layer_fn: Callable[[Layer], BackwardLayer],
    backward_bounds: List[keras.KerasTensor],
    backward_map: Optional[Dict[int, BackwardLayer]] = None,
    joint: bool = True,
    fuse: bool = True,
    output_map: Optional[Dict[int, List[keras.KerasTensor]]] = None,
    merge_layers: Optional[Layer] = None,
    fuse_layer: Optional[Layer] = None,
    **kwargs: Any,
) -> Tuple[List[keras.KerasTensor], Optional[Layer]]:
    """


    :param node:
    :param ibp:
    :param affine:
    :param input_map:
    :param layer_fn:
    :param backward_bounds:
    :param backward_map:
    :param joint:
    :param fuse:
    :return: list of 4 tensors affine upper and lower bounds
    """
    if backward_map is None:
        backward_map = {}

    if output_map is None:
        output_map = {}

    inputs = input_map[id(node)]

    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    if isinstance(node.operation, Model):
        inputs_tensors = get_disconnected_input(get_mode(ibp, affine), perturbation_domain, dtype=inputs[0].dtype)(
            inputs
        )
        _, backward_bounds, _, _ = crown_model(
            model=node.operation,
            input_tensors=inputs_tensors,
            backward_bounds=backward_bounds,
            ibp=ibp,
            affine=affine,
            perturbation_domain=None,
            finetune=False,
            joint=joint,
            fuse=False,
            **kwargs,
        )

    else:
        backward_layer = retrieve_layer(node=node, layer_fn=layer_fn, backward_map=backward_map, joint=joint)

        if id(node) not in output_map:
            backward_bounds_new = backward_layer(inputs)
            output_map[id(node)] = backward_bounds_new
        else:
            backward_bounds_new = output_map[id(node)]

            # import pdb; pdb.set_trace()
        if len(backward_bounds):
            if merge_layers is None:
                merge_layers = MergeWithPrevious(backward_bounds_new[0].shape, backward_bounds[0].shape)
            backward_bounds = merge_layers(backward_bounds_new + backward_bounds)
        else:
            backward_bounds = backward_bounds_new

    parents = node.parent_nodes

    if len(parents):
        if len(parents) > 1:
            if isinstance(backward_layer, BackwardMerge):
                raise NotImplementedError()
                crown_bound_list = []
                for backward_bound, parent in zip(backward_bounds, parents):
                    crown_bound_i, _ = crown_(
                        node=parent,
                        ibp=ibp,
                        affine=affine,
                        perturbation_domain=perturbation_domain,
                        input_map=input_map,
                        layer_fn=layer_fn,
                        backward_bounds=backward_bound,
                        backward_map=backward_map,
                        joint=joint,
                        fuse=fuse,
                    )

                    crown_bound_list.append(crown_bound_i)

                # avg_layer = Average(dtype=node.outbound_layer.dtype)
                # crown_bound = [avg_layer([e[i] for e in crown_bound_list]) for i in range(4)]
                crown_bound = crown_bound_list[0]

            else:
                raise NotImplementedError()
        else:
            crown_bound, fuse_layer_new = crown_(
                node=parents[0],
                ibp=ibp,
                affine=affine,
                perturbation_domain=perturbation_domain,
                input_map=input_map,
                layer_fn=layer_fn,
                backward_bounds=backward_bounds,
                backward_map=backward_map,
                joint=joint,
                fuse=fuse,
                output_map=output_map,
                merge_layers=None,  # AKA merge_layers
                fuse_layer=fuse_layer,
            )
            if fuse_layer is None:
                fuse_layer = fuse_layer_new
        return crown_bound, fuse_layer
    else:
        # do something
        if fuse:
            if fuse_layer is None:
                fuse_layer = Fuse(get_mode(ibp=ibp, affine=affine))
            result = fuse_layer(inputs + backward_bounds)

            return result, fuse_layer

        else:
            return backward_bounds, fuse_layer


def get_input_nodes(
    model: Model,
    dico_nodes: Dict[int, List[Node]],
    ibp: bool,
    affine: bool,
    input_tensors: List[keras.KerasTensor],
    output_map: OutputMapDict,
    layer_fn: Callable[[Layer], BackwardLayer],
    joint: bool,
    set_mode_layer: Layer,
    perturbation_domain: Optional[PerturbationDomain] = None,
    **kwargs: Any,
) -> Tuple[Dict[int, List[keras.KerasTensor]], Dict[int, BackwardLayer], Dict[int, List[keras.KerasTensor]]]:
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)
    fuse_layer = None
    input_map: Dict[int, List[keras.KerasTensor]] = {}
    backward_map: Dict[int, BackwardLayer] = {}
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    crown_map: Dict[int, List[keras.KerasTensor]] = {}
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:
            layer = node.operation

            parents = node.parent_nodes
            if not len(parents):
                # if 'debug' in kwargs.keys():
                #    import pdb; pdb.set_trace()
                input_map[id(node)] = input_tensors
            else:
                output: List[keras.KerasTensor] = []
                for parent in parents:
                    # do something
                    if id(parent) in output_map.keys():
                        output += output_map[id(parent)]
                    else:
                        output_crown, fuse_layer_tmp = crown_(
                            node=parent,
                            ibp=ibp,
                            affine=affine,
                            input_map=input_map,
                            layer_fn=layer_fn,
                            backward_bounds=[],
                            backward_map=backward_map,
                            joint=joint,
                            fuse=True,
                            perturbation_domain=perturbation_domain,
                            output_map=crown_map,
                            merge_layers=None,  # AKA merge_layers
                            fuse_layer=fuse_layer,
                        )
                        if fuse_layer is None:
                            fuse_layer = fuse_layer_tmp

                        # convert output_crown in the right mode
                        if set_mode_layer is None:
                            set_mode_layer = Convert2BackwardMode(get_mode(ibp, affine), perturbation_domain)
                        output_crown = set_mode_layer(input_tensors + output_crown)
                        output += to_list(output_crown)
                        # crown_map[id(parent)]=output_crown_

                input_map[id(node)] = output
    return input_map, backward_map, crown_map


def crown_model(
    model: Model,
    input_tensors: List[keras.KerasTensor],
    back_bounds: Optional[List[keras.KerasTensor]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    ibp: bool = True,
    affine: bool = True,
    perturbation_domain: Optional[PerturbationDomain] = None,
    finetune: bool = False,
    forward_map: Optional[OutputMapDict] = None,
    softmax_to_linear: bool = True,
    joint: bool = True,
    layer_fn: Callable[..., BackwardLayer] = to_backward,
    fuse: bool = True,
    **kwargs: Any,
) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor], Dict[int, BackwardLayer], None]:
    if back_bounds is None:
        back_bounds = []
    if forward_map is None:
        forward_map = {}
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if not isinstance(model, Model):
        raise ValueError()
    # import time
    # zero_time = time.process_time()
    has_softmax = False
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)  # do better because you modify the model eventually

    # layer_fn
    ##########
    has_iter = False
    if layer_fn is not None and len(layer_fn.__code__.co_varnames) == 1 and "layer" in layer_fn.__code__.co_varnames:
        has_iter = True

    if not has_iter:
        layer_fn_copy = deepcopy(layer_fn)

        def func(layer: Layer) -> Layer:
            return layer_fn_copy(
                layer,
                mode=get_mode(ibp, affine),
                finetune=finetune,
                perturbation_domain=perturbation_domain,
                slope=slope,
            )

        layer_fn = func

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")
    ###############

    if len(back_bounds) and len(to_list(model.output)) > 1:
        raise NotImplementedError()

    # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    # generate input_map
    if not finetune:
        joint = True
    set_mode_layer = Convert2BackwardMode(get_mode(ibp, affine), perturbation_domain)

    input_map, backward_map, crown_map = get_input_nodes(
        model=model,
        dico_nodes=dico_nodes,
        ibp=ibp,
        affine=affine,
        input_tensors=input_tensors,
        output_map=forward_map,
        layer_fn=layer_fn,
        joint=joint,
        perturbation_domain=perturbation_domain,
        set_mode_layer=set_mode_layer,
        **kwargs,
    )
    # time_1 = time.process_time()
    # print('step1', time_1-zero_time)
    # retrieve output nodes
    output = []
    output_nodes = dico_nodes[0]
    # the ordering may change
    output_names = [tensor._keras_history.operation.name for tensor in to_list(model.output)]
    fuse_layer = None
    for output_name in output_names:
        for node in output_nodes:
            if node.operation.name == output_name:
                # compute with crown
                output_crown, fuse_layer = crown_(
                    node=node,
                    ibp=ibp,
                    affine=affine,
                    input_map=input_map,
                    layer_fn=layer_fn,
                    backward_bounds=back_bounds,
                    backward_map=backward_map,
                    joint=joint,
                    fuse=fuse,
                    perturbation_domain=perturbation_domain,
                    output_map=crown_map,
                    fuse_layer=fuse_layer,
                )
                # time_2 = time.process_time()
                # print('step2', time_2-time_1)
                if fuse:
                    # import pdb; pdb.set_trace()
                    output += to_list(set_mode_layer(input_tensors + output_crown))
                else:
                    output = output_crown

    return input_tensors, output, backward_map, None


def convert_backward(
    model: Model,
    input_tensors: List[keras.KerasTensor],
    back_bounds: Optional[List[keras.KerasTensor]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    ibp: bool = True,
    affine: bool = True,
    perturbation_domain: Optional[PerturbationDomain] = None,
    finetune: bool = False,
    forward_map: Optional[OutputMapDict] = None,
    softmax_to_linear: bool = True,
    joint: bool = True,
    layer_fn: Callable[..., BackwardLayer] = to_backward,
    final_ibp: bool = True,
    final_affine: bool = False,
    **kwargs: Any,
) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor], Dict[int, BackwardLayer], None]:
    model = ensure_functional_model(model)
    if back_bounds is None:
        back_bounds = []
    if forward_map is None:
        forward_map = {}
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if len(back_bounds):
        if len(back_bounds) == 1:
            C = back_bounds[0]
            bias = K.zeros_like(C[:, 0])
            back_bounds = [C, bias] * 2
    result = crown_model(
        model=model,
        input_tensors=input_tensors,
        back_bounds=back_bounds,
        slope=slope,
        ibp=ibp,
        affine=affine,
        perturbation_domain=perturbation_domain,
        finetune=finetune,
        forward_map=forward_map,
        softmax_to_linear=softmax_to_linear,
        joint=joint,
        layer_fn=layer_fn,
        fuse=True,
        **kwargs,
    )

    input_tensors, output, backward_map, toto = result
    mode_from = get_mode(ibp, affine)
    mode_to = get_mode(final_ibp, final_affine)
    output = Convert2Mode(
        mode_from=mode_from,
        mode_to=mode_to,
        perturbation_domain=perturbation_domain,
    )(output)
    if mode_to != mode_from and mode_from == ForwardMode.IBP:
        f_input = Lambda(lambda z: Concatenate(1)([z[0][:, None], z[1][:, None]]))
        output[0] = f_input([input_tensors[1], input_tensors[0]])
    return input_tensors, output, backward_map, toto
