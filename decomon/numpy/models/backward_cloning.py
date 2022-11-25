import numpy as np
import tensorflow.keras.backend as K

from decomon.models.models import DecomonModel
from decomon.numpy.backward.layers import get_backward
from decomon.numpy.backward.utils import merge_with_previous


def crown_(
    node,
    backward_bounds=None,
    forward_init=None,
    log_bounds=None,
    log_layers=None,
    convex_domain=None,
    dico_grid=None,
    slope_grid=None,
    update_slope=False,
    reuse_slope=True,
    joint=False,
    rec=1,
):

    if backward_bounds is None:
        backward_bounds = []
    if forward_init is None:
        forward_init = []
    if log_bounds is None:
        log_bounds = {}
    if log_layers is None:
        log_layers = {}
    if convex_domain is None:
        convex_domain = {}
    if dico_grid is None:
        dico_grid = {}
    if slope_grid is None:
        slope_grid = {}
    previous = bool(len(backward_bounds))

    params_key = f"{node.outbound_layer.name}_{rec}"
    if params_key in dico_grid.keys():
        params = dico_grid[params_key]
        if params_key in slope_grid.keys():
            params.append(slope_grid[params_key])
    else:
        params = [None, None]

    if joint:
        layer_key = f"{node.outbound_layer.name}"
    else:
        layer_key = f"{node.outbound_layer.name}_{rec}"

    if layer_key in log_layers.keys():
        crown_layer = log_layers[layer_key]
        crown_layer.keras_layer = node.outbound_layer
        crown_layer.rec = rec
    else:
        crown_layer = get_backward(
            node.outbound_layer,
            previous=False,
            convex_domain=convex_domain,
            rec=rec,
            params=params,
            update_slope=update_slope,
            reuse_slope=reuse_slope,
        )
        if crown_layer.store_layer(joint, convex_domain, reuse_slope):
            log_layers[layer_key] = crown_layer

    if len(node.parent_nodes):

        if len(node.parent_nodes) > 1:
            raise NotImplementedError()
        node_ = node.parent_nodes[0]

        # step 1:
        if id(node) in log_bounds.keys():
            output = log_bounds[id(node)]
        else:

            # retrieve an input
            crown_forward, log_bounds, log_layers = crown_(
                node_,
                backward_bounds=[],
                forward_init=forward_init,
                log_bounds=log_bounds,
                log_layers=log_layers,
                convex_domain=convex_domain,
                dico_grid=dico_grid,
                slope_grid=slope_grid,
                reuse_slope=reuse_slope,
                update_slope=update_slope,
                joint=joint,
            )
            # deduce the backward bounds at the next stage
            output = crown_layer.call(crown_forward)
            if crown_layer.store_output(joint, convex_domain, reuse_slope):
                log_bounds[id(node)] = output

        if previous:
            backward_bounds = merge_with_previous(output + backward_bounds)
        else:
            backward_bounds = output

        result = crown_(
            node_,
            backward_bounds=backward_bounds,
            forward_init=forward_init,
            log_bounds=log_bounds,
            convex_domain=convex_domain,
            rec=rec + 1,
            dico_grid=dico_grid,
            slope_grid=slope_grid,
            reuse_slope=reuse_slope,
            update_slope=update_slope,
            joint=joint,
        )[0]

        if params[0] is not None:
            params_ = crown_layer.params
            for p_0, p_1 in zip(params_, params):
                K.set_value(p_1, p_0.numpy())
        return result, log_bounds, log_layers

    else:

        # input layer
        # crown_bounds
        if id(node) in log_bounds.keys() and not previous:
            crown_bounds = log_bounds[id(node)]
        else:
            crown_bounds = crown_layer.call(forward_init)
            log_bounds[id(node)] = crown_bounds

        if previous:
            backward_bounds = merge_with_previous(crown_bounds + backward_bounds)
        else:
            backward_bounds = crown_bounds

        return forward_init[:1] + backward_bounds, log_bounds, log_layers


def crown_old(
    node,
    backward_bounds=None,
    forward_init=None,
    log_bounds=None,
    convex_domain=None,
    dico_grid=None,
    slope_grid=None,
    update_slope=False,
    reuse_slope=True,
    rec=1,
):
    if backward_bounds is None:
        backward_bounds = []
    if forward_init is None:
        forward_init = []
    if log_bounds is None:
        log_bounds = {}
    if convex_domain is None:
        convex_domain = {}
    if dico_grid is None:
        dico_grid = {}
    if slope_grid is None:
        slope_grid = {}
    previous = bool(len(backward_bounds))

    params_key = f"{node.outbound_layer.name}_{rec}"
    if params_key in dico_grid.keys():
        params = dico_grid[params_key]
        if params_key in slope_grid.keys():
            params.append(slope_grid[params_key])
    else:
        params = [None, None]

    crown_layer = get_backward(
        node.outbound_layer,
        previous=previous,
        convex_domain=convex_domain,
        rec=rec,
        params=params,
        update_slope=update_slope,
        reuse_slope=reuse_slope,
    )

    # if input bounds: easy
    if len(node.parent_nodes):
        # retrieve child
        if len(node.parent_nodes) > 1:
            raise NotImplementedError()
        node_ = node.parent_nodes[0]
        if id(node_) in log_bounds.keys():
            crown_forward = log_bounds[id(node_)]
            crown_forward = forward_init[:1] + crown_forward
        else:
            crown_forward, log_bounds = crown_(
                node_,
                backward_bounds=[],
                forward_init=forward_init,
                log_bounds=log_bounds,
                convex_domain=convex_domain,
                dico_grid=dico_grid,
                slope_grid=slope_grid,
                reuse_slope=reuse_slope,
                update_slope=update_slope,
            )
            log_bounds[id(node_)] = crown_forward[-4:]

        # deduce the backward bounds at the next stage
        backward_bounds = crown_layer.call(crown_forward + backward_bounds)

        result = (
            crown_(
                node_,
                backward_bounds=backward_bounds,
                forward_init=forward_init,
                log_bounds=log_bounds,
                convex_domain=convex_domain,
                rec=rec + 1,
                dico_grid=dico_grid,
                slope_grid=slope_grid,
                reuse_slope=reuse_slope,
                update_slope=update_slope,
            )[0],
            log_bounds,
        )
        return result
    else:

        # input layer
        # crown_bounds
        if id(node) in log_bounds.keys() and not previous:
            crown_bounds = log_bounds[id(node)]
        else:
            crown_bounds = crown_layer.call(forward_init + backward_bounds)
            if not previous:
                log_bounds[id(node)] = crown_bounds

        return forward_init[:1] + crown_bounds, log_bounds


def crown(
    model,
    backward_bounds=None,
    forward_init=None,
    convex_domain=None,
    dico_grid=None,
    slope_grid=None,
    reuse_slope=False,
    update_slope=False,
    joint=False,
):
    # get nodes by depth
    if backward_bounds is None:
        backward_bounds = []
    if forward_init is None:
        forward_init = []
    if convex_domain is None:
        convex_domain = {}
    if dico_grid is None:
        dico_grid = {}
    if slope_grid is None:
        slope_grid = {}
    outputs_nodes = model._nodes_by_depth[0][0]

    if len(forward_init) == 1:
        # create the init linear relaxations on  the fly
        X = forward_init[0]
        w_f = np.repeat(np.diag([1] * X.shape[-1])[None], len(X), axis=0)
        b_f = np.zeros((len(X), X.shape[-1]))
        forward_init += [w_f, b_f] * 2
    toto = crown_(
        outputs_nodes,
        backward_bounds=backward_bounds,
        forward_init=forward_init,
        log_bounds={},
        convex_domain=convex_domain,
        dico_grid=dico_grid,
        slope_grid=slope_grid,
        reuse_slope=reuse_slope,
        update_slope=update_slope,
        joint=joint,
    )
    return toto[0]


class NumpyModel:
    def __init__(
        self,
        model,
        ibp=False,
        forward=True,
        convex_domain=None,
        has_back_bounds=False,
        dico_grid=None,
        slope_grid=None,
        update_slope=False,
        reuse_slope=False,
        joint=False,
        **kwargs,
    ):
        if convex_domain is None:
            convex_domain = {}
        if dico_grid is None:
            dico_grid = {}
        if slope_grid is None:
            slope_grid = {}
        self.IBP = ibp
        self.forward = forward
        self.convex_domain = convex_domain
        self.has_backward_bounds = has_back_bounds
        self.model = model
        self.dico_grid = dico_grid
        self.slope_grid = slope_grid
        self.reuse_slope = reuse_slope
        self.update_slope = update_slope
        self.joint = joint

    def predict(self, inputs):
        pass


class NumpyCROWNModel(NumpyModel):
    def __init__(
        self,
        model,
        ibp=False,
        forward=True,
        convex_domain=None,
        has_back_bounds=False,
        dico_grid=None,
        slope_grid=None,
        update_slope=False,
        reuse_slope=False,
        joint=False,
        **kwargs,
    ):
        super().__init__(
            model,
            ibp=ibp,
            forward=forward,
            convex_domain=convex_domain,
            has_back_bounds=has_back_bounds,
            dico_grid=dico_grid,
            slope_grid=slope_grid,
            update_slope=update_slope,
            reuse_slope=reuse_slope,
            joint=joint,
            **kwargs,
        )
        if convex_domain is None:
            convex_domain = {}
        if dico_grid is None:
            dico_grid = {}
        if slope_grid is None:
            slope_grid = {}

    def predict(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        if self.has_backward_bounds:
            backward_bounds = inputs[-4]
            inputs = inputs[:-4]
        else:
            backward_bounds = []
        return crown(
            self.model,
            backward_bounds=backward_bounds,
            forward_init=inputs,
            convex_domain=self.convex_domain,
            dico_grid=self.dico_grid,
            slope_grid=self.slope_grid,
            update_slope=self.update_slope,
            reuse_slope=self.reuse_slope,
            joint=self.joint,
        )
