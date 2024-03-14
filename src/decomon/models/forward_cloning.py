"""Module for DecomonSequential.

It inherits from keras Sequential class.

"""
from collections.abc import Callable
from typing import Any, Optional

import keras
from keras.layers import InputLayer, Layer
from keras.models import Model

from decomon.constants import Propagation, Slope
from decomon.layers import DecomonLayer
from decomon.layers.convert import to_decomon
from decomon.layers.input import ForwardInput
from decomon.layers.inputs_outputs_specs import InputsOutputsSpec
from decomon.models.utils import (
    ensure_functional_model,
    get_depth_dict,
    get_output_nodes,
    prepare_inputs_for_layer,
    wrap_outputs_from_layer_in_list,
)
from decomon.perturbation_domain import BoxDomain, PerturbationDomain


def convert_forward(
    model: Model,
    perturbation_domain_input: keras.KerasTensor,
    layer_fn: Callable[..., DecomonLayer] = to_decomon,
    slope: Slope = Slope.V_SLOPE,
    perturbation_domain: Optional[PerturbationDomain] = None,
    ibp: bool = True,
    affine: bool = True,
    **kwargs: Any,
) -> tuple[list[keras.KerasTensor], dict[int, list[keras.KerasTensor]], dict[int, DecomonLayer]]:
    """Convert keras model via forward propagation.

    Prepare layer_fn by freezing all args except layer.
    Ensure that model is functional (transform sequential ones to functional equivalent ones).

    Args:
        model: keras model to convert
        perturbation_domain_input: perturbation domain input (input to the future decomon model).
          Used to convert affine bounds into constant ones.
        layer_fn: conversion function on layers. Default to `to_decomon()`.
        slope: slope used by decomon activation layers
        perturbation_domain: perturbation domain type for keras input
        ibp: specify if constant bounds are propagated
        affine: specify if affine bounds are propagated
        **kwargs: keyword arguments to pass to layer_fn

    Returns:
        output, output_map, layer_map:
        - output: propagated bounds (concatenated), see `DecomonLayer.call()` for the format.
           output of the future decomon model
        - output_map: output of each converted node. Can be used to feed oracle bounds used by backward conversion.
        - layer_map: converted layer corresponding to each node. Can be used to transform output_map into oracle bounds

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    model = ensure_functional_model(model)
    model_input_shape = model.inputs[0].shape[1:]

    propagation = Propagation.FORWARD
    inputs_outputs_spec = InputsOutputsSpec(
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        model_input_shape=model_input_shape,
        layer_input_shape=model_input_shape,
    )

    if inputs_outputs_spec.needs_perturbation_domain_inputs():
        perturbation_domain_inputs = [perturbation_domain_input]
    else:
        perturbation_domain_inputs = []

    layer_fn_to_list = include_kwargs_layer_fn(
        layer_fn,
        model_input_shape=model_input_shape,
        slope=slope,
        perturbation_domain=perturbation_domain,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        **kwargs,
    )

    # generate input tensors
    forward_input_layer = ForwardInput(perturbation_domain=perturbation_domain, ibp=ibp, affine=affine)
    input_tensors_wo_pertubation_domain_inputs = forward_input_layer(perturbation_domain_input)

    output_map: dict[int, list[keras.KerasTensor]] = {}
    layer_list_map: dict[int, list[DecomonLayer]] = {}
    output = convert_forward_functional_model(
        model=model,
        input_tensors=input_tensors_wo_pertubation_domain_inputs,
        layer_fn=layer_fn_to_list,
        common_inputs_part=perturbation_domain_inputs,
        output_map=output_map,
        layer_map=layer_list_map,
    )
    layer_map: dict[int, DecomonLayer] = {k: v[0] for k, v in layer_list_map.items()}

    return output, output_map, layer_map


def convert_forward_functional_model(
    model: Model,
    layer_fn: Callable[[Layer], list[Layer]],
    input_tensors: list[keras.KerasTensor],
    common_inputs_part: Optional[list[keras.KerasTensor]] = None,
    output_map: Optional[dict[int, list[keras.KerasTensor]]] = None,
    layer_map: Optional[dict[int, list[Layer]]] = None,
    submodel: bool = False,
) -> list[keras.KerasTensor]:
    """Convert a functional keras model via forward propagation.

    Used
    - for decomon conversion in forward mode with layer_fn=lambda l: [to_decomon(l)]
    - also for preprocessing keras models (e.g. splitting activation layers) with layer_fn=preprocess_layer

    Hypothesis: potential embedded submodels have only one input and one output.

    Args:
        model: keras model to convert
        layer_fn: callable converting a layer into a list of converted layers
        input_tensors: input tensors used by the converted (list of) layer(s) corresponding to the keras model input nodes
        common_inputs_part: inputs part common to all converted layers.
            The inputs of a converted layers is
            the concatenation of the outputs of the converted layers corresponding to its inputs nodes
            + this common_inputs_part (only once even for merging layers)
            This allows to pass perturbation_domain_inputs only once when creating the decomon forward version of a keras model.
            Default to an empty list.
        output_map: output of each converted node (to be used when called recursively on submodels)
            When a layer is converted into several layers, we store the propagated output of the last one.
            This map is updated during the conversion.
        layer_map: map between node and converted layers (to be used when called recursively on submodels)
            This map is updated during the conversion.
        submodel: specify if called from within another conversion to propagate through an embedded submodel

    Returns:
        concatenated outputs of the converted layers corresponding to the output nodes of the keras model

    """
    if common_inputs_part is None:
        common_inputs_part = []

    # ensure the model is functional (to be able to use get_depth_dict), convert sequential ones into functional ones
    model = ensure_functional_model(model)

    # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    if output_map is None:
        output_map = {}
    if layer_map is None:
        layer_map = {}
    output: list[keras.KerasTensor] = input_tensors
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:
            layer = node.operation
            parents = node.parent_nodes
            if id(node) in output_map.keys():
                continue
            if len(parents):
                output = []
                for parent in parents:
                    output += output_map[id(parent)]

            if isinstance(layer, InputLayer):
                # no conversion, propagate output unchanged
                pass
            elif isinstance(layer, Model):
                output = convert_forward_functional_model(
                    model=layer,
                    input_tensors=output,
                    common_inputs_part=common_inputs_part,
                    layer_fn=layer_fn,
                    output_map=output_map,
                    layer_map=layer_map,
                )
            else:
                if id(node) in layer_map:
                    # avoid converting twice layers that are shared by several nodes
                    converted_layers = layer_map[id(node)]
                else:
                    converted_layers = layer_fn(layer)
                    layer_map[id(node)] = converted_layers
                for converted_layer in converted_layers:
                    output = wrap_outputs_from_layer_in_list(
                        converted_layer(prepare_inputs_for_layer(output + common_inputs_part))
                    )
            output_map[id(node)] = wrap_outputs_from_layer_in_list(output)

    output = []
    output_nodes = get_output_nodes(model)
    if submodel and len(output_nodes) > 1:
        raise NotImplementedError(
            "convert_forward_functional_model() not yet implemented for model "
            "whose embedded submodels have multiple outputs."
        )
    for node in output_nodes:
        output += output_map[id(node)]

    return output


def include_kwargs_layer_fn(
    layer_fn: Callable[..., Layer],
    model_input_shape: tuple[int, ...],
    perturbation_domain: PerturbationDomain,
    ibp: bool,
    affine: bool,
    propagation: Propagation,
    slope: Slope,
    **kwargs: Any,
) -> Callable[[Layer], list[Layer]]:
    """Include external parameters in the function converting layers

    In particular, include propagation=Propagation.FORWARD.

    Args:
        layer_fn:
        model_input_shape:
        perturbation_domain:
        ibp:
        affine:
        slope:
        **kwargs:

    Returns:

    """

    def func(layer: Layer) -> list[Layer]:
        return [
            layer_fn(
                layer,
                model_input_shape=model_input_shape,
                slope=slope,
                perturbation_domain=perturbation_domain,
                ibp=ibp,
                affine=affine,
                propagation=propagation,
                **kwargs,
            )
        ]

    return func
