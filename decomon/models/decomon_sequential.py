"""Module for MonotonicSequential.

It inherits from keras Sequential class.

"""
from __future__ import absolute_import
import inspect
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from ..layers.decomon_layers import to_monotonic
from ..layers.core import Box
from tensorflow.keras.layers import InputLayer, Input, Layer
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list
from copy import deepcopy
import numpy as np
from decomon.layers.utils import softmax_to_linear, get_upper, get_lower


def include_dim_layer_fn(layer_fn, input_dim, dc_decomp=False, grad_bounds=False, convex_domain={}, n_subgrad=0):
    """
    include external parameters inside the translation of a layer to its decomon counterpart
    :param layer_fn:
    :param input_dim:
    :param dc_decomp:
    :param grad_bounds:
    :param convex_domain:
    :param n_subgrad
    :return:
    """
    if input_dim <= 0:
        if n_subgrad and "n_subgrad" in inspect.signature(layer_fn).parameters:
            layer_fn_copy = deepcopy(layer_fn)

            def func(x):
                return layer_fn_copy(
                    x,
                    input_dim,
                    dc_decomp=dc_decomp,
                    grad_bounds=grad_bounds,
                    convex_domain=convex_domain,
                    n_subgrad=n_subgrad,
                )

            layer_fn = func
        else:
            return layer_fn
    else:
        if "input_dim" in inspect.signature(layer_fn).parameters:
            layer_fn_copy = deepcopy(layer_fn)
            if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
                input_dim = (2, input_dim)
            else:
                if convex_domain["name"] == Box.name and not isinstance(input_dim, tuple):
                    input_dim = (2, input_dim)

            def func(x):
                return layer_fn_copy(
                    x,
                    input_dim,
                    dc_decomp=dc_decomp,
                    grad_bounds=grad_bounds,
                    convex_domain=convex_domain,
                    n_subgrad=n_subgrad,
                )

            layer_fn = func

        else:

            warnings.warn(
                "Expected {} to have an input dim option. " "Henceworth the to_monotonic function will be used instead"
            )
            if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
                input_dim = (2, input_dim)
            else:
                if convex_domain["name"] == Box.name and not isinstance(input_dim, tuple):
                    input_dim = (2, input_dim)

            def func(x):
                return to_monotonic(
                    x,
                    input_dim=input_dim,
                    dc_decomp=dc_decomp,
                    grad_bounds=grad_bounds,
                    convex_domain=convex_domain,
                    n_subgrad=n_subgrad,
                )

            layer_fn = func

    return layer_fn


def clone(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=-1,
    dc_decomp=False,
    grad_bounds=False,
    convex_domain={},
    n_subgrad=0,
):
    """
    :param model: Keras model
    :param input_tensors: List of input tensors to be used as inputs of our model or None
    :param layer_fn: cloning function to translate a layer into its decomon decomposition
    :param input_dim: the input dimension use to propagate our linear relaxation
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param n_subgrad: integer
    :return: a decomon model
    """

    if grad_bounds:
        raise NotImplementedError()

    if isinstance(model, Sequential):
        return clone_sequential_model(
            model,
            input_tensors,
            layer_fn,
            input_dim=input_dim,
            dc_decomp=dc_decomp,
            grad_bounds=grad_bounds,
            convex_domain=convex_domain,
            n_subgrad=n_subgrad,
        )

    return clone_functional_model(
        model,
        input_tensors,
        layer_fn,
        input_dim=input_dim,
        dc_decomp=dc_decomp,
        grad_bounds=grad_bounds,
        convex_domain=convex_domain,
        n_subgrad=n_subgrad,
    )


def clone_sequential_model(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=-1,
    dc_decomp=False,
    grad_bounds=False,
    convex_domain={},
    n_subgrad=0,
):
    """Clone a `Sequential` model instance.
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
    :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param n_subgrad: integer
    :return: An instance of `Sequential` reproducing the behavior of the original model with decomon layers.
    :raises: ValueError: in case of invalid `model` argument value or `layer_fn` argument value.
    """
    layer_fn = include_dim_layer_fn(
        layer_fn,
        input_dim,
        dc_decomp=dc_decomp,
        grad_bounds=grad_bounds,
        convex_domain=convex_domain,
        n_subgrad=n_subgrad,
    )

    model = softmax_to_linear(model)

    if grad_bounds:
        raise NotImplementedError()

    if not isinstance(model, Sequential):
        raise ValueError(
            "Expected `model` argument " "to be a `Sequential` model instance, " "but got:",
            model,
        )

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
                list_layer = []
                for layer_ in layer._layers:
                    list_layer += _get_layer(layer_)
                return list_layer
            return [clone_functional_model(layer, layer_fn=layer_fn)]

        if isinstance(layer, Layer):
            return [layer_fn(layer)]
        raise ValueError

    monotonic_layers = []
    for layer in model._layers:
        monotonic_layers += _get_layer(layer)

    if input_tensors is None:
        # init input_tensors
        # import pdb; pdb.set_trace()
        # shape = model.get_input_shape_at(0)[1:]

        input_shape = list(K.int_shape(model.input)[1:])

        if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
            input_dim_ = (2, input_dim)
            z_tensor = Input(input_dim_)
        elif convex_domain["name"] == Box.name and not isinstance(input_dim, tuple):
            input_dim_ = (2, input_dim)
            z_tensor = Input(input_dim_)
        else:

            if isinstance(input_dim, tuple):
                z_tensor = Input(input_dim)
            else:
                z_tensor = Input((input_dim,))

        # x_min_tensor = Input((input_dim,))
        # x_max_tensor = Input((input_dim,))
        y_tensor = Input(tuple(input_shape))
        w_u_tensor = Input(tuple([input_dim] + input_shape))
        w_l_tensor = Input(tuple([input_dim] + input_shape))
        h_tensor = Input(tuple(input_shape))
        g_tensor = Input(tuple(input_shape))
        b_u_tensor = Input(tuple(input_shape))
        b_l_tensor = Input(tuple(input_shape))

        u_c_tensor = Input(tuple(input_shape))
        l_c_tensor = Input(tuple(input_shape))

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
        if dc_decomp:
            input_tensors += [h_tensor, g_tensor]

    else:
        # assert that input_tensors is a List of 6 InputLayer objects
        # If input tensors are provided, the original model's InputLayer is
        # overwritten with a different InputLayer.
        assert isinstance(input_tensors, list), "expected input_tensors to be a List or None, but got dtype={}".format(
            input_tensors.dtype
        )

        if dc_decomp:
            assert len(input_tensors) == 10, "wrong number of inputs, expexted 10 but got {}".format(len(input_tensors))
        else:
            assert len(input_tensors) == 8, "wrong number of inputs, expexted 10 but got {}".format(len(input_tensors))
        assert min(
            [isinstance(input_tensor_i, InputLayer) for input_tensor_i in input_tensors]
        ), "expected a list of InputLayer"

    # apply the list of monotonic layers:
    output = input_tensors

    for layer in monotonic_layers:
        output = layer(output)

    # return Model(input_tensors, output)
    return DecomonModel(input_tensors, output, convex_domain=convex_domain)


def clone_functional_model(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=1,
    dc_decomp=False,
    grad_bounds=False,
    convex_domain={},
    n_subgrad=0,
):
    """

    :param model:
    :param input_tensors: optional list of input tensors to build the model upon. If not provided, placeholder will
    be created.
    :param layer_fn: callable to be applied on non-input layers in the model. By default it clones the layer.
    :param input_dim: the input dimension use to propagate our linear relaxation
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param n_subgrad: integer
    :return:
    """

    if grad_bounds:
        raise NotImplementedError()

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)
    if isinstance(model, Sequential):
        raise ValueError(
            "Expected `model` argument " "to be a functional `Model` instance, " "got a `Sequential` instance instead:",
            model,
        )

    layer_fn = include_dim_layer_fn(
        layer_fn,
        input_dim,
        dc_decomp=dc_decomp,
        grad_bounds=grad_bounds,
        convex_domain=convex_domain,
        n_subgrad=n_subgrad,
    )
    model = softmax_to_linear(model)

    # we only handle one input
    assert len(model._input_layers) == 1, "error: Expected one input only but got {}".format(len(model._input_layers))

    def clone(layer):
        return layer.__class__.from_config(layer.get_config())

    # if convex:
    #    layer_fn = lambda x:to_monotonic(x, convex=True)

    layer_map = {}  # Cache for created layers.
    tensor_map = {}  # Map {reference_tensor: (corresponding_tensor, mask)}

    if input_tensors is None:
        layer = model._input_layers[0]
        input_shape_vec = layer.input_shape[0]
        input_shape = tuple(list(input_shape_vec)[1:])

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
        w_u_tensor = Input(shape=input_shape_w, dtype=layer.dtype, name="w_u_" + layer.name)
        w_l_tensor = Input(shape=input_shape_w, dtype=layer.dtype, name="w_l_" + layer.name)
        b_u_tensor = Input(shape=input_shape, dtype=layer.dtype, name="b_u_" + layer.name)
        b_l_tensor = Input(shape=input_shape, dtype=layer.dtype, name="b_l_" + layer.name)
        u_c_tensor = Input(shape=input_shape, dtype=layer.dtype, name="u_c_" + layer.name)
        l_c_tensor = Input(shape=input_shape, dtype=layer.dtype, name="l_c_" + layer.name)

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
        if dc_decomp:
            h_tensor = Input(shape=input_shape, dtype=layer.dtype, name="h_" + layer.name)
            g_tensor = Input(shape=input_shape, dtype=layer.dtype, name="g_" + layer.name)
            input_tensors += [h_tensor, g_tensor]

        # Cache newly created input layer.
        newly_created_input_layers = [input_tensor._keras_history[0] for input_tensor in input_tensors]
        layer_map[layer] = newly_created_input_layers

    else:
        # Make sure that all input tensors come from a Keras layer.
        # If tensor comes from an input layer: cache the input layer.
        input_tensors = to_list(input_tensors)
        _input_tensors = []
        names_i = ["y", "z", "u_c", "w_u", "b_u", "l_c", "w_l", "b_l"]
        if dc_decomp:
            names_i += ["h", "g"]
            # names_i = ['h', 'g', 'x_min', 'x_max', 'u_c', 'w_u', 'b_u', 'l_c', 'w_l', 'b_l']
            assert len(input_tensors) == 10, "error, Expected 10 input tensors but got {}".format(len(input_tensors))
        else:
            assert len(input_tensors) == 8, "error, Expected 8 input tensors but got {}".format(len(input_tensors))

        for i, x in enumerate(input_tensors):

            if not K.is_keras_tensor(x):
                name = model._input_layers[0].name
                input_tensor = Input(tensor=x, name=name + "_{}".format(names_i[i]))
                _input_tensors.append(input_tensor)
                # Cache newly created input layer.
                original_input_layer = x._keras_history[0]
                newly_created_input_layer = input_tensor._keras_history[0]
                if original_input_layer not in layer_map.keys():
                    layer_map[original_input_layer] = newly_created_input_layer
                else:
                    if isinstance(layer_map[original_input_layer], list):
                        layer_map[original_input_layer] += [newly_created_input_layer]
                    else:
                        layer_map[original_input_layer] = [layer_map[original_input_layer]]
                        layer_map[original_input_layer] += [newly_created_input_layer]
            else:
                _input_tensors.append(x)
        input_tensors = _input_tensors

    tensor_map[id(model.inputs[0])] = (input_tensors, None)

    # Iterated over every node in the reference model, in depth order.
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            # Recover the corresponding layer.
            layer_ = node.outbound_layer
            # Get or create layer.
            if layer_ not in layer_map:
                # Clone layer.
                if isinstance(layer_, Model):
                    if isinstance(layer_, Sequential):
                        new_layer = clone_sequential_model(layer_, layer_fn=layer_fn)
                    else:
                        new_layer = clone_functional_model(layer_, layer_fn=layer_fn)
                else:
                    new_layer = layer_fn(layer_)

                layer_map[layer_] = new_layer
                layer = new_layer  # ????
            else:
                # Reuse previously cloned layer.
                layer = layer_map[layer_]
                # Don't call InputLayer multiple times.
                if isinstance(layer_, InputLayer):
                    continue

            # Gather inputs to call the new layer.
            reference_input_tensors = node.input_tensors
            reference_output_tensors = node.output_tensors

            if not isinstance(reference_input_tensors, list):
                reference_input_tensors = [reference_input_tensors]

            if not isinstance(reference_output_tensors, list):
                reference_output_tensors = [reference_output_tensors]

            # If all previous input tensors are available in tensor_map,
            # then call node.inbound_layer on them.
            computed_data = []  # List of tuples (input, mask).
            for x in reference_input_tensors:
                if id(x) in tensor_map:
                    # import pdb; pdb.set_trace()
                    computed_data.append(tensor_map[id(x)])
            if len(computed_data) == len(reference_input_tensors):

                # Call layer.
                if hasattr(node, "arguments"):
                    kwargs = node.arguments
                else:
                    kwargs = {}
                if len(computed_data) == 1:
                    computed_tensor, computed_mask = computed_data[0]  # (list of) tensors, None

                    if has_arg(layer_.call, "mask"):
                        if "mask" not in kwargs:
                            kwargs["mask"] = computed_mask

                    # are we sure that layer is Monotonic ?
                    output_tensors = to_list(layer(computed_tensor, **kwargs))

                    if not (isinstance(output_tensors[0], list)):
                        output_tensors = [output_tensors]

                    if layer_.supports_masking:
                        output_masks = [layer.compute_mask(computed_tensor, computed_mask)]
                    else:
                        output_masks = [None] * len(output_tensors)

                    computed_tensors = [computed_tensor]
                    computed_masks = [computed_mask]

                else:
                    computed_tensors = []
                    for x in computed_data:
                        computed_tensors += x[0]
                    computed_masks = [x[1] for x in computed_data]
                    if has_arg(layer.call, "mask"):
                        if "mask" not in kwargs:
                            kwargs["mask"] = computed_masks
                    output_tensors = to_list(layer(computed_tensors, **kwargs))
                    if not (isinstance(output_tensors[0], list)):
                        output_tensors = [output_tensors]

                    if layer.supports_masking:
                        output_masks = to_list(layer.compute_mask(computed_tensors, computed_masks))
                    else:
                        output_masks = [None] * len(output_tensors)

                # Update tensor_map.
                for x, y, mask in zip(reference_output_tensors, output_tensors, output_masks):
                    tensor_map[id(x)] = (y, mask)

    # Check that we did compute the model outputs,
    # then instantiate a new model from inputs and outputs.

    output_tensors = []
    for x in model.outputs:
        assert id(x) in tensor_map, "Could not compute output " + str(x)
        tensor, _ = tensor_map[id(x)]
        if isinstance(tensor, list):
            for elem in tensor:
                output_tensors.append(elem)
        else:
            output_tensors.append(tensor)

    return DecomonModel(input_tensors, output_tensors, convex_domain=convex_domain)


def convert(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    dc_decomp=False,
    grad_bounds=False,
    convex_domain={},
    n_subgrad=0,
):
    """

    :param model: Keras model
    :param input_tensors: input_tensors: List of input tensors to be used as inputs of our model or None
    :param layer_fn: layer_fn: cloning function to translate a layer into its decomon decomposition
    :param dc_decomp: dc_decomp: boolean that indicates whether we return a
    difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: convex_domain: the type of convex domain
    :param n_subgrad: integer for optimizing linear bounds
    :return: a decomon model
    """

    # raise NotImplementedError()

    if grad_bounds:
        raise NotImplementedError()

    input_dim = np.prod(model.input_shape[1:])

    if input_tensors is None:

        x = model.input
        input_shape = tuple(model.input_shape[1:])
        if isinstance(input_dim, tuple):
            input_dim_ = list(input_dim)[-1]
        else:
            input_dim_ = input_dim

        if len(convex_domain) == 0:
            input_shape_x = (2, input_dim_)
        elif convex_domain["name"] == Box.name and not isinstance(input_dim, tuple):
            input_shape_x = (2, input_dim)
        else:
            if isinstance(input_dim, tuple):
                input_shape_x = input_dim
            else:
                input_shape_x = (input_dim_,)

        input_shape_w = tuple([input_dim_] + list(input_shape))

        z_tensor = Input(shape=input_shape_x, dtype=x.dtype, name="z_0")
        y_tensor = Input(shape=input_shape, dtype=x.dtype, name="y_0")

        input_tensors = [y_tensor, z_tensor]

        if dc_decomp:
            h_tensor = Input(shape=input_shape, dtype=x.dtype, name="h_0")
            g_tensor = Input(shape=input_shape, dtype=x.dtype, name="g_0")
            input_tensors += [h_tensor, g_tensor]

        def get_input(input_):

            if dc_decomp:
                y_tensor, z_tensor, h_tensor, g_tensor = input_
            else:
                y_tensor, z_tensor = input_

            # create b_u, b_l, u_c, l_c from the previous tensors (with a lambda layer)
            b_u_tensor = K.zeros_like(y_tensor)
            b_l_tensor = K.zeros_like(y_tensor)

            toto = K.reshape(
                K.cast(tf.linalg.diag([1.0] * input_dim_), y_tensor.dtype),
                input_shape_w,
            )[None]
            w_u_tensor = K.expand_dims(K.zeros_like(y_tensor), 1) + toto
            w_l_tensor = K.expand_dims(K.zeros_like(y_tensor), 1) + toto

            # compute upper and lower bound
            l_c_tensor = get_lower(z_tensor, w_l_tensor, b_l_tensor, convex_domain=convex_domain)
            u_c_tensor = get_upper(z_tensor, w_u_tensor, b_u_tensor, convex_domain=convex_domain)

            output = [
                y_tensor,
                z_tensor,
                u_c_tensor,
                w_l_tensor,
                b_u_tensor,
                l_c_tensor,
                w_l_tensor,
                b_l_tensor,
            ]
            if dc_decomp:
                output += [h_tensor, g_tensor]

            return output

        # create a custom layer that hides the formulation of linear relaxation
        lambda_layer = Lambda(get_input)
    else:
        lambda_layer = Lambda(lambda x: x)

    input_tensors_ = lambda_layer(input_tensors)

    model_monotonic = clone(
        model,
        input_tensors=None,
        layer_fn=layer_fn,
        input_dim=input_dim,
        dc_decomp=dc_decomp,
        grad_bounds=grad_bounds,
        convex_domain=convex_domain,
        n_subgrad=n_subgrad,
    )

    output = model_monotonic(input_tensors_)

    return DecomonModel(input_tensors, output, convex_domain=convex_domain)


def set_domain_priv(convex_domain_prev, convex_domain):
    msg = "we can only change the parameters of the convex domain, not its nature"

    convex_domain_ = convex_domain
    if convex_domain == {}:
        convex_domain = {"name": Box.name}

    if len(convex_domain_prev) == 0 or convex_domain_prev["name"] == Box.name:
        # Box
        if convex_domain["name"] != Box.name:
            raise NotImplementedError(msg)

    if convex_domain_prev["name"] != convex_domain["name"]:
        raise NotImplementedError(msg)

    return convex_domain_


class DecomonModel(tf.keras.Model):
    def __init__(self, input, output, convex_domain={}, optimize="True", **kwargs):
        super(DecomonModel, self).__init__(input, output, **kwargs)
        self.convex_domain = convex_domain
        self.optimize = optimize

    def set_domain(self, convex_domain):
        set_domain_priv(self.convex_domain, convex_domain)
        """
        msg = "we can only change the parameters of the convex domain, not its nature"

        convex_domain_ = convex_domain
        if convex_domain == {}:
            convex_domain = {"name": Box.name}

        if len(self.convex_domain) == 0 or self.convex_domain["name"] == Box.name:
            # Box
            if convex_domain["name"] != Box.name:
                raise NotImplementedError(msg)

        if self.convex_domain["name"] != convex_domain["name"]:
            raise NotImplementedError(msg)

        self.convex_domain = convex_domain_
        """


class DecomonSequential(tf.keras.Sequential):
    def __init__(self, layers=None, convex_domain={}, optimize="False", name=None, **kwargs):
        super(DecomonSequential, self).__init__(layers=layers, name=name, **kwargs)
        self.convex_domain = convex_domain
        self.optimize = optimize

    def set_domain(self, convex_domain):
        set_domain_priv(self.convex_domain, convex_domain)
