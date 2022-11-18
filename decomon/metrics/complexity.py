from __future__ import absolute_import

from tensorflow.keras.layers import InputLayer, Lambda
from tensorflow.python.keras.utils.generic_utils import to_list


def get_graph_complexity(model):

    # do not consider Input nodes or Lambda nodes
    # enumerate the number of
    nb_nodes = 0
    depth_keys = list(model._nodes_by_depth.keys())

    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            if isinstance(node.outbound_layer, (InputLayer, Lambda)):
                continue

            nb_nodes += 1

    return nb_nodes
