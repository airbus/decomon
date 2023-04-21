from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, InputLayer, Lambda, Layer
from tensorflow.keras.models import Model

from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.crown import Convert2Mode
from decomon.layers.decomon_layers import to_decomon
from decomon.models.backward_cloning import convert_backward
from decomon.models.forward_cloning import (
    LayerMapDict,
    OutputMapDict,
    convert_forward,
    convert_forward_functional_model,
)
from decomon.models.models import DecomonModel
from decomon.models.utils import (
    ConvertMethod,
    get_input_dim,
    get_input_tensors_keras_only,
    get_mode,
    split_activation,
)
from decomon.utils import ConvexDomainType, Slope, get_lower, get_upper


class FeedDirection(Enum):
    FORWARD = "feed_forward"
    BACKWARD = "feed_backward"


def get_direction(method: Union[str, ConvertMethod]) -> FeedDirection:
    if ConvertMethod(method) in [ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_AFFINE, ConvertMethod.FORWARD_HYBRID]:
        return FeedDirection.FORWARD
    else:
        return FeedDirection.BACKWARD


def get_ibp_affine_from_method(method: Union[str, ConvertMethod]) -> Tuple[bool, bool]:
    method = ConvertMethod(method)
    if method in [ConvertMethod.FORWARD_IBP, ConvertMethod.CROWN_FORWARD_IBP]:
        return True, False
    if method in [ConvertMethod.FORWARD_AFFINE, ConvertMethod.CROWN_FORWARD_AFFINE]:
        return False, True
    if method in [ConvertMethod.FORWARD_HYBRID, ConvertMethod.CROWN_FORWARD_HYBRID]:
        return True, True
    if method == ConvertMethod.CROWN:
        return True, False
    return True, True


def switch_mode_mapping(forward_map: OutputMapDict, ibp: bool, affine: bool, method: Union[str, ConvertMethod]) -> None:
    raise NotImplementedError()


def split_activations_in_keras_model(
    model: Model,
    input_shape: Optional[Tuple[int, ...]] = None,
    input_tensors: Optional[List[tf.Tensor]] = None,
) -> Model:
    if input_shape is None:
        try:
            model_input_shape = model.input_shape
        except:
            raise ValueError("model.input_shape must be defined if input_shape argument is None")
        else:
            input_shape = model_input_shape[1:]

    if input_tensors is None:
        input_tensors = get_input_tensors_keras_only(
            model=model,
            input_shape=input_shape,
        )

    _, output, _, _ = convert_forward_functional_model(
        model=model,
        input_tensors=input_tensors,
        softmax_to_linear=False,
        layer_fn=split_activation,
    )

    return Model(
        inputs=input_tensors,
        outputs=output,
    )


# create status
def convert(
    model: Model,
    input_tensors: List[tf.Tensor],
    method: Union[str, ConvertMethod] = ConvertMethod.CROWN,
    ibp: bool = False,
    affine: bool = False,
    back_bounds: Optional[List[tf.Tensor]] = None,
    layer_fn: Callable[..., Layer] = to_decomon,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    input_dim: int = -1,
    convex_domain: Optional[Dict[str, Any]] = None,
    finetune: bool = False,
    forward_map: Optional[OutputMapDict] = None,
    shared: bool = True,
    softmax_to_linear: bool = True,
    finetune_forward: bool = False,
    finetune_backward: bool = False,
    final_ibp: bool = False,
    final_affine: bool = False,
    **kwargs: Any,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], Union[LayerMapDict, Dict[int, BackwardLayer]], Optional[OutputMapDict],]:

    if back_bounds is None:
        back_bounds = []
    if convex_domain is None:
        convex_domain = {}
    if finetune:
        finetune_forward = True
        finetune_backward = True

    if isinstance(method, str):
        method = ConvertMethod(method.lower())

    # prepare the Keras Model: split non-linear activation functions into separate Activation layers
    model = split_activations_in_keras_model(model)

    layer_map: Union[LayerMapDict, Dict[int, BackwardLayer]]

    if method != ConvertMethod.CROWN:

        ibp_, affine_ = ibp, affine

        input_tensors, output, layer_map, forward_map = convert_forward(
            model=model,
            input_tensors=input_tensors,
            layer_fn=layer_fn,
            slope=slope,
            input_dim=input_dim,
            dc_decomp=False,
            convex_domain=convex_domain,
            ibp=ibp_,
            affine=affine_,
            finetune=finetune_forward,
            shared=shared,
            softmax_to_linear=softmax_to_linear,
            back_bounds=back_bounds,
        )

    if get_direction(method) == FeedDirection.BACKWARD:
        input_tensors, output, layer_map, forward_map = convert_backward(
            model=model,
            input_tensors=input_tensors,
            back_bounds=back_bounds,
            slope=slope,
            input_dim=input_dim,
            convex_domain=convex_domain,
            ibp=ibp,
            affine=affine,
            finetune=finetune_backward,
            forward_map=forward_map,
            final_ibp=final_ibp,
            final_affine=final_affine,
            **kwargs,
        )
    else:

        # check final_ibp and final_affine
        mode_from = get_mode(ibp, affine)
        mode_to = get_mode(final_ibp, final_affine)
        output = Convert2Mode(
            mode_from=mode_from, mode_to=mode_to, convex_domain=convex_domain, dtype=model.layers[0].dtype
        )(output)

    # build decomon model
    return input_tensors, output, layer_map, forward_map


def clone(
    model: Model,
    layer_fn: Callable[..., Layer] = to_decomon,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    convex_domain: Optional[Dict[str, Any]] = None,
    method: Union[str, ConvertMethod] = ConvertMethod.CROWN,
    back_bounds: Optional[List[tf.Tensor]] = None,
    finetune: bool = False,
    shared: bool = True,
    finetune_forward: bool = False,
    finetune_backward: bool = False,
    extra_inputs: Optional[List[tf.Tensor]] = None,
    to_keras: bool = True,
    final_ibp: Optional[bool] = None,
    final_affine: Optional[bool] = None,
    **kwargs: Any,
) -> DecomonModel:

    if convex_domain is None:
        convex_domain = {}
    if back_bounds is None:
        back_bounds = []
    if extra_inputs is None:
        extra_inputs = []
    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    ibp_, affine_ = get_ibp_affine_from_method(method)
    if final_ibp is None:
        final_ibp = ibp_
    if final_affine is None:
        final_affine = affine_

    if isinstance(method, str):
        method = ConvertMethod(method.lower())

    if len(convex_domain) and isinstance(convex_domain["name"], str):
        convex_domain["name"] = ConvexDomainType(convex_domain["name"].lower())

    if not to_keras:
        raise NotImplementedError("Only convert to Keras for now.")

    if finetune:
        finetune_forward = True
        finetune_backward = True

    input_dim: int = np.prod(model.input_shape[1:])
    input_shape = None
    input_shape_vec = None

    for input_layer in model._input_layers:
        if len(input_layer.input_shape) > 1:
            raise ValueError(f"Expected one input tensor but got {len(input_layer.input_shape)}")
        input_shape_vec_ = input_layer.input_shape[0]
        input_shape_ = tuple(list(input_shape_vec_)[1:])

        if input_shape_vec is None:
            input_shape_vec = input_shape_vec_
        if input_shape is None:
            input_shape = input_shape_
        else:
            if not np.allclose(input_shape, input_shape_):
                raise ValueError("Expected that every input layers use the same input_tensor")

    input_shape_x: Tuple[int, ...]
    if len(convex_domain) == 0 or convex_domain["name"] != ConvexDomainType.BALL:
        input_shape_x = (2, input_dim)
    else:
        input_shape_x = (input_dim,)

    z_tensor = Input(shape=input_shape_x, dtype=model.layers[0].dtype)

    if len(convex_domain) == 0 or convex_domain["name"] != ConvexDomainType.BALL:

        if ibp_:
            u_c_tensor = Lambda(lambda z: z[:, 1], dtype=z_tensor.dtype)(z_tensor)
            l_c_tensor = Lambda(lambda z: z[:, 0], dtype=z_tensor.dtype)(z_tensor)

        if affine_:
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
            if ibp_:
                l_c_tensor = Lambda(
                    lambda var: var - K.cast(radius, dtype=model.layers[0].dtype), dtype=model.layers[0].dtype
                )(z_tensor)
            if affine_:
                z_value = K.cast(0.0, model.layers[0].dtype)
                o_value = K.cast(1.0, model.layers[0].dtype)

                W = tf.linalg.diag(z_value * u_c_tensor + o_value)
                b = z_value * u_c_tensor

        else:
            z_value = K.cast(0.0, model.layers[0].dtype)
            o_value = K.cast(1.0, model.layers[0].dtype)

            def get_bounds(z: tf.Tensor) -> List[tf.Tensor]:
                output = []
                W = tf.linalg.diag(z_value * z + o_value)
                b = z_value * z
                if affine_:
                    output += [W, b]
                if ibp_:
                    u_c_ = get_upper(z, W, b, convex_domain)
                    l_c_ = get_lower(z, W, b, convex_domain)
                    output += [u_c_, l_c_]
                return output

            output_ = get_bounds(z_tensor)
            if ibp_:
                u_c_tensor, l_c_tensor = output_[-2:]
            if affine_:
                W, b = output_[:2]

    if ibp_ and affine_:
        input_tensors = [z_tensor] + [u_c_tensor, W, b] + [l_c_tensor, W, b]
    elif ibp_ and not affine_:
        input_tensors = [u_c_tensor, l_c_tensor]
    elif not ibp_ and affine_:
        input_tensors = [z_tensor] + [W, b] + [W, b]
    else:
        raise NotImplementedError("not ibp and not affine not implemented")

    _, output, _, _ = convert(
        model,
        layer_fn=layer_fn,
        slope=slope,
        input_tensors=input_tensors,
        back_bounds=back_bounds,
        method=method,
        ibp=ibp_,
        affine=affine_,
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
        final_affine=final_affine,
    )

    back_bounds_ = []
    for elem in back_bounds:
        if isinstance(elem._keras_history.layer, InputLayer):
            back_bounds_.append(elem)

    return DecomonModel(
        inputs=[z_tensor] + back_bounds_ + extra_inputs,
        outputs=output,
        convex_domain=convex_domain,
        dc_decomp=False,
        method=method,
        ibp=final_ibp,
        affine=final_affine,
        finetune=finetune,
        shared=shared,
        backward_bounds=(len(back_bounds) > 0),
    )
