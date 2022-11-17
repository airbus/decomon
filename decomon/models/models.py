from __future__ import absolute_import

import inspect
import warnings

import tensorflow as tf

from ..layers.core import Box, StaticVariables


# create static variables for varying convex domain
class Backward:
    name = "backward"


class Forward:
    name = "forward"


class DecomonModel(tf.keras.Model):
    def __init__(
        self,
        input,
        output,
        convex_domain=None,
        dc_decomp=False,
        method=Forward.name,
        optimize="True",
        IBP=True,
        forward=True,
        finetune=False,
        shared=True,
        backward_bounds=False,
        **kwargs,
    ):
        super(DecomonModel, self).__init__(input, output, **kwargs)
        if convex_domain is None:
            convex_domain = {}
        self.convex_domain = convex_domain
        self.optimize = optimize
        self.nb_tensors = StaticVariables(dc_decomp).nb_tensors
        self.dc_decomp = dc_decomp
        self.method = method
        self.IBP = IBP
        self.forward = forward
        self.finetune = finetune
        self.backward_bounds = backward_bounds

    def set_domain(self, convex_domain):
        convex_domain = set_domain_priv(self.convex_domain, convex_domain)
        self.convex_domain = convex_domain
        for layer in self.layers:
            if hasattr(layer, "convex_domain"):
                layer.convex_domain = self.convex_domain

    def freeze_weights(self):
        for layer in self.layers:
            if hasattr(layer, "freeze_weights"):
                layer.freeze_weights()

    def unfreeze_weights(self):
        for layer in self.layers:
            if hasattr(layer, "unfreeze_weights"):
                layer.unfreeze_weights()

    def freeze_alpha(self):
        for layer in self.layers:
            if hasattr(layer, "freeze_alpha"):
                layer.freeze_alpha()

    def unfreeze_alpha(self):
        for layer in self.layers:
            if hasattr(layer, "unfreeze_alpha"):
                layer.unfreeze_alpha()

    def reset_finetuning(self):
        for layer in self.layers:
            if hasattr(layer, "reset_finetuning"):
                layer.reset_finetuning()


class DecomonSequential(tf.keras.Sequential):
    def __init__(
        self,
        layers=None,
        convex_domain=None,
        dc_decomp=False,
        mode=Forward.name,
        optimize="False",
        IBP=True,
        forward=False,
        finetune=False,
        **kwargs,
    ):
        super(DecomonSequential, self).__init__(layers=layers, name=name, **kwargs)
        if convex_domain is None:
            convex_domain = {}
        self.convex_domain = convex_domain
        self.optimize = optimize
        self.nb_tensors = StaticVariables(dc_decomp).nb_tensors
        self.dc_decomp = dc_decomp
        self.mode = mode
        self.IBP = IBP
        self.forward = forward

    def set_domain(self, convex_domain):
        convex_domain = set_domain_priv(self.convex_domain, convex_domain)
        self.convex_domain = convex_domain
        for layer in self.layers:
            if hasattr(layer, "convex_domain"):
                layer.convex_domain = self.convex_domain

    def freeze_weights(self):
        for layer in self.layers:
            if hasattr(layer, "freeze_weights"):
                layer.freeze_weights()

    def unfreeze_weights(self):
        for layer in self.layers:
            if hasattr(layer, "unfreeze_weights"):
                layer.unfreeze_weights()

    def freeze_alpha(self):
        for layer in self.layers:
            if hasattr(layer, "freeze_alpha"):
                layer.freeze_alpha()

    def unfreeze_alpha(self):
        for layer in self.layers:
            if hasattr(layer, "unfreeze_alpha"):
                layer.unfreeze_alpha()

    def reset_finetuning(self):
        for layer in self.layers:
            if hasattr(layer, "reset_finetuning"):
                layer.unfreeze_alpha()


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
