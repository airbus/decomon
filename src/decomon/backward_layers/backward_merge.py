from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Wrapper

from decomon.backward_layers.utils import (
    backward_add,
    backward_maximum,
    backward_minimum,
    backward_multiply,
    backward_substract,
    get_identity_lirpa,
)
from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.decomon_merge_layers import (
    DecomonAdd,
    DecomonConcatenate,
    DecomonDot,
    DecomonMaximum,
    DecomonMinimum,
    DecomonMultiply,
    DecomonSubtract,
)
from decomon.layers.utils import broadcast, multiply, permute_dimensions, split
from decomon.utils import Slope


class BackwardMerge(ABC, Wrapper):

    layer: Layer
    _trainable_weights: List[tf.Variable]

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        kwargs.pop("slope", None)
        kwargs.pop("finetune", None)
        super().__init__(layer, **kwargs)
        self.rec = rec
        if isinstance(self.layer, DecomonLayer):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
            self.dc_decomp = self.layer.dc_decomp
        else:
            self.mode = ForwardMode(mode)
            if convex_domain is None:
                self.convex_domain = {}
            else:
                self.convex_domain = convex_domain
            self.dc_decomp = dc_decomp

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "rec": self.rec,
                "mode": self.mode,
                "convex_domain": self.convex_domain,
                "dc_decomp": self.dc_decomp,
            }
        )
        return config

    @abstractmethod
    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:
        """
        Args:
            inputs

        Returns:

        """
        pass

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape

        Returns:

        """
        # generic case: nothing to do before call
        pass

    def freeze_weights(self) -> None:
        pass

    def unfreeze_weights(self) -> None:
        pass

    def freeze_alpha(self) -> None:
        pass

    def unfreeze_alpha(self) -> None:
        pass

    def reset_finetuning(self) -> None:
        pass


class BackwardAdd(BackwardMerge):
    """Backward  LiRPA of Add"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.op = DecomonAdd(mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp).call

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:

        w_out_u, b_out_u, w_out_l, b_out_l = get_identity_lirpa(inputs)
        n_comp = 2
        if self.mode == ForwardMode.AFFINE:
            n_comp = 5
        if self.mode == ForwardMode.HYBRID:
            n_comp = 7

        n_elem = len(inputs) // n_comp

        if n_elem == 1:
            return [[w_out_u, b_out_u, w_out_l, b_out_l]]
        else:
            if n_elem > 2:
                raise NotImplementedError()
            else:

                inputs_0 = inputs[:n_comp]
                inputs_1 = inputs[n_comp:]

                bounds_0, bounds_1 = backward_add(
                    inputs_0,
                    inputs_1,
                    w_out_u,
                    b_out_u,
                    w_out_l,
                    b_out_l,
                    convex_domain=self.convex_domain,
                    mode=self.mode,
                )
                return [bounds_0, bounds_1]


class BackwardAverage(BackwardMerge):
    """Backward  LiRPA of Average"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.op = DecomonAdd(mode=self.mode, convex_domain=self.convex_domain, dc_decomp=False).call

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:

        w_out_u, b_out_u, w_out_l, b_out_l = get_identity_lirpa(inputs)

        n_comp = 2
        if self.mode == ForwardMode.AFFINE:
            n_comp = 5
        if self.mode == ForwardMode.HYBRID:
            n_comp = 7

        n_elem = len(inputs) // n_comp
        if n_elem == 1:
            return [[w_out_u, b_out_u, w_out_l, b_out_l]]
        else:
            bounds: List[List[tf.Tensor]] = []
            input_bounds: List[List[tf.Tensor]] = []

            for j in np.arange(1, n_elem)[::-1]:
                inputs_1 = inputs[n_comp * j : n_comp * (j + 1)]
                if j == 1:
                    inputs_0 = inputs[:n_comp]
                else:
                    inputs_0 = self.op(inputs[: n_comp * j])
                if len(bounds) == 0:
                    bounds_0, bounds_1 = backward_add(
                        inputs_0,
                        inputs_1,
                        w_out_u,
                        b_out_u,
                        w_out_l,
                        b_out_l,
                        convex_domain=self.convex_domain,
                        mode=self.mode,
                    )
                else:
                    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = bounds[-1]
                    bounds_0, bounds_1 = backward_add(
                        inputs_0,
                        inputs_1,
                        w_out_u_,
                        b_out_u_,
                        w_out_l_,
                        b_out_l_,
                        convex_domain=self.convex_domain,
                        mode=self.mode,
                    )

                input_bounds.append(bounds_1)
                bounds.append(bounds_0)
                if j == 1:
                    input_bounds.append(bounds_0)

            input_bounds = input_bounds[::-1]
            return [[1.0 / n_elem * elem_i for elem_i in elem] for elem in input_bounds]


class BackwardSubtract(BackwardMerge):
    """Backward  LiRPA of Subtract"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonSubtract):
            raise KeyError()

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == ForwardMode.AFFINE:
            n_comp = 6
        if self.mode == ForwardMode.HYBRID:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_substract(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardMaximum(BackwardMerge):
    """Backward  LiRPA of Maximum"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonMaximum):
            raise KeyError()

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == ForwardMode.AFFINE:
            n_comp = 6
        if self.mode == ForwardMode.HYBRID:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_maximum(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardMinimum(BackwardMerge):
    """Backward  LiRPA of Minimum"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonMinimum):
            raise KeyError()

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == ForwardMode.AFFINE:
            n_comp = 6
        if self.mode == ForwardMode.HYBRID:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_minimum(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardConcatenate(BackwardMerge):
    """Backward  LiRPA of Concatenate"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonConcatenate):
            raise KeyError()

        self.axis = self.layer.axis

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:
        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == ForwardMode.AFFINE:
            n_comp = 6
        if self.mode == ForwardMode.HYBRID:
            n_comp = 8

        n_elem = len(x_) // n_comp
        n_list = [inputs[n_comp * i : n_comp * (i + 1)][0].shape[self.axis] for i in range(len(x_) // n_comp)]
        axis_w = self.axis
        if axis_w != -1:
            axis_w += 1
        w_out_u_list = tf.split(w_out_u, n_list, axis_w)
        w_out_l_list = tf.split(w_out_l, n_list, axis_w)
        b_out_u_list = tf.split(b_out_u, n_list, self.axis)
        b_out_l_list = tf.split(b_out_l, n_list, self.axis)

        bounds = [[w_out_u_list[i], b_out_u_list[i], w_out_l_list[i], b_out_l_list[i]] for i in range(n_elem)]

        return bounds


class BackwardMultiply(BackwardMerge):
    """Backward  LiRPA of Multiply"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonMultiply):
            raise KeyError()

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == ForwardMode.AFFINE:
            n_comp = 6
        if self.mode == ForwardMode.HYBRID:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_multiply(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardDot(BackwardMerge):
    """Backward  LiRPA of Dot"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonDot):
            raise KeyError()

        self.axes = [i for i in self.layer.axes]
        self.op = BackwardAdd(self.layer)

        raise NotImplementedError()

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[List[tf.Tensor]]:

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == ForwardMode.AFFINE:
            n_comp = 6
        if self.mode == ForwardMode.HYBRID:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        # permute dimensions and reshape
        inputs_0 = inputs_list[0]
        inputs_1 = inputs_list[1]

        n_0 = len(inputs_0[0].shape) - 2
        n_1 = len(inputs_1[0].shape) - 2

        input_0_0 = permute_dimensions(inputs_0, self.axes[0], mode=self.mode)
        input_1_0 = permute_dimensions(inputs_1, self.axes[1], mode=self.mode)

        inputs_0_ = broadcast(input_0_0, n_1, -1, mode=self.mode)
        inputs_1_ = broadcast(input_1_0, n_0, 2, mode=self.mode)

        inputs_ = multiply(
            inputs_0_,
            inputs_1_,
            dc_decomp=self.layer.dc_decomp,
            convex_domain=self.convex_domain,
            mode=self.mode,
        )

        inputs_add_ = split(inputs_, axis=self.axes[0], mode=self.mode)
        inputs_add = []
        for elem in inputs_add_:
            inputs_add += elem

        bounds = self.op.call(inputs_add + inputs[-4:])
        n = len(inputs_add)
        bounds_ = [[1.0 / n * elem for elem in bounds_i] for bounds_i in bounds]

        # concatenate
        reshape_ = [-1, 1] + list(inputs_[0].shape[1:]) + list(w_out_u.shape[3:])

        bounds_reshape = [
            [K.reshape(elem[0], reshape_), elem[1], K.reshape(elem[2], reshape_), elem[3]] for elem in bounds_
        ]
        w_u_ = K.concatenate([elem[0] for elem in bounds_reshape], 1)
        b_u_ = sum([elem[1] for elem in bounds_reshape], 1)
        w_l_ = K.concatenate([elem[2] for elem in bounds_reshape], 1)
        b_l_ = sum([elem[3] for elem in bounds_reshape])

        bounds_m_0, bounds_m_1 = backward_multiply(
            inputs_0_,
            inputs_1_,
            w_u_,
            b_u_,
            w_l_,
            b_l_,
            convex_domain=self.convex_domain,
            mode=self.mode,
        )

        shape_0 = [-1, 1] + list(input_0_0[0].shape[1:]) + list(w_u_.shape[3:])
        shape_1 = [-1, 1] + list(input_1_0[0].shape[1:]) + list(w_u_.shape[3:])

        bounds_m_0 = [
            K.reshape(bounds_m_0[0], shape_0),
            bounds_m_0[1],
            K.reshape(bounds_m_0[2], shape_0),
            bounds_m_0[3],
        ]
        bounds_m_1 = [
            K.reshape(bounds_m_1[0], shape_1),
            bounds_m_1[1],
            K.reshape(bounds_m_1[2], shape_1),
            bounds_m_1[3],
        ]

        axes = [i for i in self.axes]
        if axes[0] == -1:
            axes[0] = len(inputs_0[0].shape)
        if axes[1] == -1:
            axes[1] = len(inputs_1[0].shape)

        index_0 = np.arange(len(shape_0))
        index_0[2] = axes[0] + 1
        index_0[axes[0] + 1] = 2

        index_1 = np.arange(len(shape_1))
        index_1[2] = axes[1] + 1
        index_1[axes[1] + 1] = 2

        bounds_m_0 = [
            K.permute_dimensions(bounds_m_0[0], index_0),
            bounds_m_0[1],
            K.permute_dimensions(bounds_m_0[2], index_0),
            bounds_m_0[3],
        ]
        bounds_m_1 = [
            K.permute_dimensions(bounds_m_1[0], index_1),
            bounds_m_1[1],
            K.permute_dimensions(bounds_m_1[2], index_1),
            bounds_m_1[3],
        ]

        return [bounds_m_0, bounds_m_1]
