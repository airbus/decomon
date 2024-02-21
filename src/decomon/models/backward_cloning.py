from copy import deepcopy
from collections.abc import Callable
from typing import Any, Optional, Union

import keras
import keras.ops as K
from keras.config import floatx
from keras.layers import Lambda, Layer
from keras.models import Model
from keras.src.ops.node import Node
from keras.src.utils.python_utils import to_list

from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    Propagation,
    Slope,
    get_affine,
    get_mode,
)
from decomon.layers import DecomonLayer
from decomon.layers.convert import to_decomon
from decomon.layers.merging.base_merge import DecomonMerge
from decomon.models.crown import Convert2BackwardMode, Fuse, MergeWithPrevious
from decomon.models.utils import (
    ensure_functional_model,
    get_depth_dict,
    get_output_nodes,
)
from decomon.types import Tensor


def crown(
    node: Node,
    layer_fn: Callable[[Layer, tuple[int, ...]], DecomonLayer],
    model_output_shape: tuple[int, ...],
    backward_bounds: list[keras.KerasTensor],
    backward_map: dict[int, DecomonLayer],
    oracle_map: dict[int, Union[list[keras.KerasTensor], list[list[keras.KerasTensor]]]],
    forward_output_map: dict[int, list[keras.KerasTensor]],
    forward_layer_map: dict[int, DecomonLayer],
    crown_output_map: dict[int, list[keras.KerasTensor]],
    perturbation_domain_input: keras.KerasTensor,
    perturbation_domain: PerturbationDomain,
) -> list[keras.KerasTensor]:
    """

    Args:
        node: node of the model until which the backward bounds have been propagated
        layer_fn: function converting a keras layer into its backward version,
            according to the proper (sub)model output shape
            Will be passed to `get_oracle()` for crown oracle deriving from sub-crowns
        model_output_shape: model_output_shape of the current crown,
            to be passed to `crown_model()` on embedded submodels.
        backward_bounds: backward bounds to propagate
        oracle_map: oracle bounds on inputs of each keras layer, stored by layers id
        oracle_map: already registered oracle bounds per node
            To be used by `get_oracle()`.
        forward_output_map: forward outputs per node from a previously performed forward conversion.
            To be used by `get_oracle()`.
        forward_layer_map: forward decomon layer per node from a previously performed forward conversion.
            To be used by `get_oracle()`.
        crown_output_map: output of subcrowns per output node.
            Avoids relaunching a crown if several nodes share parents.
            To be used by `get_oracle()`.
        backward_map: stores converted layer by node for the current crown
          (should depend on the proper model output and thus change for each sub-crown)

    Returns:
        propagated backward bounds until model input for the proper output node

    """
    parents = node.parent_nodes
    if len(parents) == 0:
        # Input layer => no conversion, propagate output unchanged
        return backward_bounds
    else:
        if isinstance(node.operation, Model):
            # TO CHECK perturbation input submodel ?
            # forward oracle:
            #  - ibp (+-affine): not needed
            #  - affine (w/o ibp): forward_output give affine bounds on outer model input => perturbation_domain
            # crown oracle:
            #  - see where stop subcrowns, maybe to submodel inputs,
            #    in which case we need to construct perturbation_domain_input_submodel from get_oracle(node)
            perturbation_domain_input_submodel = perturbation_domain_input
            backward_bounds = crown_model(
                model=node.operation,
                layer_fn=layer_fn,
                backward_bounds=[backward_bounds],
                perturbation_domain_input=perturbation_domain_input_submodel,
                perturbation_domain=perturbation_domain,
                oracle_map=oracle_map,
                forward_output_map=forward_output_map,
                forward_layer_map=forward_layer_map,
                crown_output_map=crown_output_map,
                is_submodel=True,
                backward_map=backward_map,
                model_output_shape=model_output_shape,
            )
        else:
            if id(node) in backward_map:
                backward_layer = backward_map[id(node)]
            else:
                backward_layer = layer_fn(node.operation, model_output_shape)
                backward_map[id(node)] = backward_layer

            # get oracle bounds if needed
            if backward_layer.inputs_outputs_spec.needs_oracle_bounds():
                constant_oracle_bounds = get_oracle(
                    node=node,
                    perturbation_domain_input=perturbation_domain_input,
                    perturbation_domain=perturbation_domain,
                    oracle_map=oracle_map,
                    forward_output_map=forward_output_map,
                    forward_layer_map=forward_layer_map,
                    backward_layer=backward_layer,
                    crown_output_map=crown_output_map,
                    layer_fn=layer_fn,
                )

            else:
                constant_oracle_bounds = []

            # propagate backward bounds through the decomon layer
            backward_layer_inputs = backward_layer.inputs_outputs_spec.flatten_inputs(
                affine_bounds_to_propagate=backward_bounds,
                constant_oracle_bounds=constant_oracle_bounds,
                perturbation_domain_inputs=[],
            )
            backward_layer_outputs = backward_layer(backward_layer_inputs)
            backward_bounds, _ = backward_layer.inputs_outputs_spec.split_outputs(backward_layer_outputs)

        # Call crown recursively on parent nodes
        if isinstance(backward_layer, DecomonMerge):
            # merging layer
            crown_bounds_list: list[list[Tensor]] = []
            for backward_bounds_i, parent in zip(backward_bounds, parents):
                crown_bounds_list.append(
                    crown(
                        node=parent,
                        layer_fn=layer_fn,
                        model_output_shape=model_output_shape,
                        backward_bounds=backward_bounds_i,
                        backward_map=backward_map,
                        oracle_map=oracle_map,
                        forward_output_map=forward_output_map,
                        forward_layer_map=forward_layer_map,
                        crown_output_map=crown_output_map,
                        perturbation_domain_input=perturbation_domain_input,
                        perturbation_domain=perturbation_domain,
                    )
                )
            # reduce by summing all bounds together
            # (indeed all bounds are partial affine bounds on model output w.r.t the same model input
            #  under the hypotheses of a single model input)
            crown_bounds = backward_layer.inputs_outputs_spec.sum_backward_bounds(crown_bounds_list)

        elif len(parents) > 1:
            raise RuntimeError("Node with multiple parents should have been converted to a DecomonMerge layer.")
        else:
            # unary layer
            crown_bounds = crown(
                node=parents[0],
                layer_fn=layer_fn,
                model_output_shape=model_output_shape,
                backward_bounds=backward_bounds,
                backward_map=backward_map,
                oracle_map=oracle_map,
                forward_output_map=forward_output_map,
                forward_layer_map=forward_layer_map,
                crown_output_map=crown_output_map,
                perturbation_domain_input=perturbation_domain_input,
                perturbation_domain=perturbation_domain,
            )
        return crown_bounds


def get_oracle(
    node: Node,
    perturbation_domain_input: keras.KerasTensor,
    perturbation_domain: PerturbationDomain,
    oracle_map: dict[int, Union[list[keras.KerasTensor], list[list[keras.KerasTensor]]]],
    forward_output_map: dict[int, list[keras.KerasTensor]],
    forward_layer_map: dict[int, DecomonLayer],
    backward_layer: DecomonLayer,
    crown_output_map: dict[int, list[keras.KerasTensor]],
    layer_fn: Callable[[Layer, tuple[int, ...]], DecomonLayer],
) -> Union[list[keras.KerasTensor], list[list[keras.KerasTensor]]]:
    """Get oracle bounds "on demand".

    When needed by a node, get oracle constant bounds on keras layer inputs either:
    - from `oracle_map`, if already computed
    - from forward oracle, when a first forward conversion has been done
    - from crown oracle, by launching sub-crowns on parent nodes

    Args:
        node: considered node whose operation (keras layer) needs oracle bounds
        perturbation_domain_input: perturbation domain input
        perturbation_domain: perturbation domain type on keras model input
        oracle_map: already registered oracle bounds per node
        forward_output_map: forward outputs per node from a previously performed forward conversion.
            To be used for forward oracle.
        forward_layer_map: forward decomon layer per node from a previously performed forward conversion.
            To be used for forward oracle.
        backward_layer: converted backward decomon layer for this node.
            To be used for crown oracle.
        crown_output_map: output of subcrowns per output node.
            Avoids relaunching a crown if several nodes share parents.
            To be used for crown oracle.
        layer_fn: callable converting a layer and a model_output_shape into a (backward) decomon layer.
            To be used for crown oracle.

    Returns:
        oracle bounds on node inputs

    """
    # Do not recompute if already existing
    if id(node) in oracle_map:
        return oracle_map[id(node)]

    parents = node.parent_nodes
    if not len(parents):
        # input node: can be deduced directly from perturbation domain
        oracle_bounds = [
            perturbation_domain.get_lower_x(x=perturbation_domain_input),
            perturbation_domain.get_upper_x(x=perturbation_domain_input),
        ]
    else:
        if id(node) in forward_layer_map:
            # forward oracle
            forward_layer = forward_layer_map[id(node)]
            forward_input: list[keras.KerasTensor] = []
            for parent in parents:
                forward_input += forward_output_map[id(parent)]
            if forward_layer.inputs_outputs_spec.needs_perturbation_domain_inputs():
                forward_input += [perturbation_domain_input]
            oracle_bounds = forward_layer.call_oracle(forward_input)
        else:
            # crown oracle
            crown_bounds = []
            for parent in parents:
                if id(parent) in crown_output_map:
                    # already computed sub-crown?
                    crown_bounds_parent = crown_output_map[id(parent)]
                else:
                    submodel_output_shape = get_model_output_shape(node=parent, backward_bounds_node=[])
                    crown_bounds_parent = crown(
                        node=parent,
                        layer_fn=layer_fn,
                        model_output_shape=submodel_output_shape,
                        backward_bounds=[],
                        backward_map={},  # new output node, thus new backward_map
                        oracle_map=oracle_map,
                        forward_output_map=forward_output_map,
                        forward_layer_map=forward_layer_map,
                        crown_output_map=crown_output_map,
                        perturbation_domain_input=perturbation_domain_input,
                        perturbation_domain=perturbation_domain,
                    )
                    # store sub-crown output
                    crown_output_map[id(parent)] = crown_bounds_parent
                crown_bounds += crown_bounds_parent

            # deduce oracle bounds from affine bounds on keras layer inputs
            backward_oracle_inputs = crown_bounds + [perturbation_domain_input]
            oracle_bounds = backward_layer.call_oracle(backward_oracle_inputs)

    # store oracle
    oracle_map[id(node)] = oracle_bounds

    return oracle_bounds


def crown_model(
    model: Model,
    layer_fn: Callable[[Layer, tuple[int, ...]], DecomonLayer],
    backward_bounds: list[list[keras.KerasTensor]],
    perturbation_domain_input: keras.KerasTensor,
    perturbation_domain: PerturbationDomain,
    oracle_map: Optional[dict[int, Union[list[keras.KerasTensor], list[list[keras.KerasTensor]]]]] = None,
    forward_output_map: Optional[dict[int, list[keras.KerasTensor]]] = None,
    forward_layer_map: Optional[dict[int, DecomonLayer]] = None,
    crown_output_map: Optional[dict[int, list[keras.KerasTensor]]] = None,
    is_submodel: bool = False,
    backward_map: Optional[dict[int, DecomonLayer]] = None,
    model_output_shape: Optional[tuple[int, ...]] = None,
) -> list[keras.KerasTensor]:
    """Convert a functional keras model via crown algorithm (backward propagation)

    Hypothesis: potential embedded submodels have only one input and one output.

    Args:
        model: keras model to convert
        layer_fn: callable converting a layer and a model_output_shape into a (backward) decomon layer
        perturbation_domain_input: perturbation domain input
        perturbation_domain: perturbation domain type on keras model input
        backward_bounds: should be of the same size as the number of model outputs
            (each sublist potentially empty for starting with identity bounds)
        oracle_map: already registered oracle bounds per node
        forward_output_map: forward outputs per node from a previously performed forward conversion.
            To be used for forward oracle.
        forward_layer_map: forward decomon layer per node from a previously performed forward conversion.
            To be used for forward oracle.
        crown_output_map: output of subcrowns per output node.
            Avoids relaunching a crown if several nodes share parents.
            To be used for crown oracle.
        is_submodel: specify if called from within a crown to propagate through an embedded submodel
        backward_map: stores converted layer by node for the current crown
          Should be set only if is_submodel is True.
        model_output_shape: if submodel is True, must be set to the output_shape used in the current crown

    Returns:
        concatenated propagated backward bounds corresponding to each output node of the keras model

    """
    if oracle_map is None:
        oracle_map = {}
    if forward_layer_map is None:
        forward_layer_map = {}
    if forward_output_map is None:
        forward_output_map = {}
    if crown_output_map is None:
        crown_output_map = {}

    # Retrieve output nodes in same order as model.outputs
    output_nodes = get_output_nodes(model)
    if is_submodel and len(output_nodes) > 1:
        raise NotImplementedError(
            "crown_model() not yet implemented for model " "whose embedded submodels have multiple outputs."
        )
    # Apply crown on each output, with the appropriate backward_bounds and model_output_shape
    output = []
    for node, backward_bounds_node in zip(output_nodes, backward_bounds):
        if is_submodel:
            # for embedded submodel, pass the frozen model_output_shape fixed and the current backward_map
            backward_map_node = backward_map
            if model_output_shape is None:
                raise RuntimeError("`submodel_output_shape` must be set if `submodel` is True.")
        else:
            # new backward_map and new model_output_shape for each output node
            model_output_shape = get_model_output_shape(node=node, backward_bounds_node=backward_bounds_node)
            backward_map_node = {}

        output_crown = crown(
            node=node,
            layer_fn=layer_fn,
            model_output_shape=model_output_shape,
            backward_bounds=backward_bounds_node,
            backward_map=backward_map_node,
            oracle_map=oracle_map,
            forward_output_map=forward_output_map,
            forward_layer_map=forward_layer_map,
            crown_output_map=crown_output_map,
            perturbation_domain_input=perturbation_domain_input,
            perturbation_domain=perturbation_domain,
        )
        output += output_crown

    return output


def convert_backward(
    model: Model,
    perturbation_domain_input: keras.KerasTensor,
    perturbation_domain: Optional[PerturbationDomain] = None,
    layer_fn: Callable[..., DecomonLayer] = to_decomon,
    backward_bounds: Optional[list[list[keras.KerasTensor]]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    forward_output_map: Optional[dict[int, list[keras.KerasTensor]]] = None,
    forward_layer_map: Optional[dict[int, DecomonLayer]] = None,
    **kwargs: Any,
) -> list[keras.KerasTensor]:
    """Convert keras model via backward propagation.

    Prepare layer_fn by freezing all args except layer and model_output_shape.
    Ensure that model is functional (transform sequential ones to functional equivalent ones).

    Args:
        model: keras model to convert
        perturbation_domain_input: perturbation domain input
        perturbation_domain: perturbation domain type on keras model input
        layer_fn: callable converting a layer and a model_output_shape into a (backward) decomon layer
        backward_bounds: if set, should be of the same size as the number of model outputs
            (each sublist potentially empty for starting with identity bounds)
        forward_output_map: forward outputs per node from a previously performed forward conversion.
            To be used for forward oracle if not empty.
        forward_layer_map: forward decomon layer per node from a previously performed forward conversion.
            To be used for forward oracle if not empty.
        slope: slope used by decomon activation layers
        **kwargs: keyword arguments to pass to layer_fn

    Returns:
        propagated affine bounds (concatenated), output of the future decomon model

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if backward_bounds is None:
        backward_bounds = [[]] * len(model.outputs)

    model = ensure_functional_model(model)
    propagation = Propagation.BACKWARD

    layer_fn = include_kwargs_layer_fn(
        layer_fn,
        slope=slope,
        perturbation_domain=perturbation_domain,
        propagation=propagation,
        **kwargs,
    )

    output = crown_model(
        model=model,
        layer_fn=layer_fn,
        backward_bounds=backward_bounds,
        perturbation_domain_input=perturbation_domain_input,
        perturbation_domain=perturbation_domain,
        forward_output_map=forward_output_map,
        forward_layer_map=forward_layer_map,
    )

    return output


def get_model_output_shape(node: Node, backward_bounds_node: list[Tensor]):
    """Get outer model output shape w/o batchsize.

    If any backward bounds are passed, we deduce the outer keras model output shape from it.
    We assume for that:
    - backward_bounds = [w_l, b_l, w_u, b_u]
    - we can have w_l, w_u in diagonal representation (w_l.shape == b_l.shape)
    - we have the batchsize included in the backward_bounds

    => model_output_shape = backward_bounds[1].shape[1:]

    If no backward bounds are given, we fall back to the output shape of the given output node.

    Args:
        node: current output node of the (potentially inner) keras model to convert
        backward_bounds_node: backward bounds specified for this node

    Returns:
        outer keras model output shape, excluding batchsize

    """
    if len(backward_bounds_node) == 0:
        return node.outputs[0].shape[1:]
    else:
        _, b, _, _ = backward_bounds_node
        return b.shape[1:]


def include_kwargs_layer_fn(
    layer_fn: Callable[..., DecomonLayer],
    perturbation_domain: PerturbationDomain,
    propagation: Propagation,
    slope: Slope,
    **kwargs: Any,
) -> Callable[[Layer, tuple[int, ...]], DecomonLayer]:
    """Include external parameters in the function converting layers

    In particular, include propagation=Propagation.BACKWARD.

    Args:
        layer_fn:
        perturbation_domain:
        propagation:
        slope:
        **kwargs:

    Returns:

    """

    def func(layer: Layer, model_output_shape: tuple[int, ...]) -> DecomonLayer:
        return layer_fn(
            layer,
            model_output_shape=model_output_shape,
            slope=slope,
            perturbation_domain=perturbation_domain,
            propagation=propagation,
            **kwargs,
        )

    return func
