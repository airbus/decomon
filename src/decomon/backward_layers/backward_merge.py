from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
import numpy as np
from keras.layers import Layer, Wrapper

from decomon.backward_layers.utils import (
    backward_add,
    backward_maximum,
    backward_minimum,
    backward_multiply,
    backward_subtract,
    get_identity_lirpa,
)
from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    get_affine,
    get_ibp,
)
from decomon.layers.core import DecomonLayer
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


class BackwardMerge(ABC, Wrapper):
    layer: Layer
    _trainable_weights: List[keras.Variable]

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        kwargs.pop("slope", None)
        kwargs.pop("finetune", None)
        super().__init__(layer, **kwargs)
        self.rec = rec
        if isinstance(self.layer, DecomonLayer):
            self.mode = self.layer.mode
            self.perturbation_domain = self.layer.perturbation_domain
            self.dc_decomp = self.layer.dc_decomp
        else:
            self.mode = ForwardMode(mode)
            if perturbation_domain is None:
                self.perturbation_domain = BoxDomain()
            else:
                self.perturbation_domain = perturbation_domain
            self.dc_decomp = dc_decomp
        self.inputs_outputs_spec = InputsOutputsSpec(
            dc_decomp=self.dc_decomp, mode=self.mode, perturbation_domain=self.perturbation_domain
        )

    @property
    def ibp(self) -> bool:
        return get_ibp(self.mode)

    @property
    def affine(self) -> bool:
        return get_affine(self.mode)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "rec": self.rec,
                "mode": self.mode,
                "perturbation_domain": self.perturbation_domain,
                "dc_decomp": self.dc_decomp,
            }
        )
        return config

    @abstractmethod
    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        """
        Args:
            inputs

        Returns:

        """
        pass

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
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
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.op = DecomonAdd(
            mode=self.mode, perturbation_domain=self.perturbation_domain, dc_decomp=self.dc_decomp
        ).call

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        #  check number of inputs
        if len(inputs_list) == 1:  # nothing to merge
            return [[w_u_out, b_u_out, w_l_out, b_l_out]]
        elif len(inputs_list) == 2:
            bounds_0, bounds_1 = backward_add(
                inputs_list[0],
                inputs_list[1],
                w_u_out,
                b_u_out,
                w_l_out,
                b_l_out,
                perturbation_domain=self.perturbation_domain,
                mode=self.mode,
                dc_decomp=self.dc_decomp,
            )
            return [bounds_0, bounds_1]

        else:
            raise NotImplementedError("This layer is intended to merge only 2 layers.")


class BackwardAverage(BackwardMerge):
    """Backward  LiRPA of Average"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        self.op = DecomonAdd(mode=self.mode, perturbation_domain=self.perturbation_domain, dc_decomp=False).call

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        n_elem = len(inputs_list)
        #  check number of inputs
        if n_elem == 1:  # nothing to merge
            return [[w_u_out, b_u_out, w_l_out, b_l_out]]
        else:
            bounds: List[List[keras.KerasTensor]] = []
            input_bounds: List[List[keras.KerasTensor]] = []

            for j in range(n_elem - 1, 0, -1):
                inputs_1 = inputs_list[j]
                if j == 1:
                    inputs_0 = inputs_list[0]
                else:
                    inputs_0 = self.op(list(chain(*inputs_list[: (j - 1)])))  # merge (j-1) first inputs
                bounds_0, bounds_1 = backward_add(
                    inputs_0,
                    inputs_1,
                    w_u_out,
                    b_u_out,
                    w_l_out,
                    b_l_out,
                    perturbation_domain=self.perturbation_domain,
                    mode=self.mode,
                    dc_decomp=self.dc_decomp,
                )
                input_bounds.append(bounds_1)
                bounds.append(bounds_0)
                if j == 1:
                    input_bounds.append(bounds_0)
                # update bounds to use for next iteration
                w_u_out, b_u_out, w_l_out, b_l_out = bounds_0

            input_bounds = input_bounds[::-1]
            return [[1.0 / n_elem * elem_i for elem_i in elem] for elem in input_bounds]


class BackwardSubtract(BackwardMerge):
    """Backward  LiRPA of Subtract"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonSubtract):
            raise KeyError()

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        n_elem = len(inputs_list)

        if n_elem != 2:
            raise NotImplementedError("This layer is intended to merge only 2 layers.")

        return backward_subtract(
            inputs_list[0],
            inputs_list[1],
            w_u_out,
            b_u_out,
            w_l_out,
            b_l_out,
            perturbation_domain=self.layer.perturbation_domain,
            mode=self.mode,
            dc_decomp=self.dc_decomp,
        )


class BackwardMaximum(BackwardMerge):
    """Backward  LiRPA of Maximum"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonMaximum):
            raise KeyError()

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        n_elem = len(inputs_list)

        if n_elem != 2:
            raise NotImplementedError("This layer is intended to merge only 2 layers.")

        return backward_maximum(
            inputs_list[0],
            inputs_list[1],
            w_u_out,
            b_u_out,
            w_l_out,
            b_l_out,
            perturbation_domain=self.layer.perturbation_domain,
            mode=self.mode,
            dc_decomp=self.dc_decomp,
        )


class BackwardMinimum(BackwardMerge):
    """Backward  LiRPA of Minimum"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonMinimum):
            raise KeyError()

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        n_elem = len(inputs_list)

        if n_elem != 2:
            raise NotImplementedError("This layer is intended to merge only 2 layers.")

        return backward_minimum(
            inputs_list[0],
            inputs_list[1],
            w_u_out,
            b_u_out,
            w_l_out,
            b_l_out,
            perturbation_domain=self.layer.perturbation_domain,
            mode=self.mode,
            dc_decomp=self.dc_decomp,
        )


class BackwardConcatenate(BackwardMerge):
    """Backward  LiRPA of Concatenate"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonConcatenate):
            raise KeyError()

        self.axis = self.layer.axis

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        n_elem = len(inputs_list)
        n_list = [self.inputs_outputs_spec.get_kerasinputshape(subinputs)[self.axis] for subinputs in inputs_list]
        indices_split = np.cumsum(n_list[:-1])
        axis_w = self.axis
        if axis_w != -1:
            axis_w += 1
        w_u_out_list = K.split(w_u_out, indices_split, axis_w)
        w_l_out_list = K.split(w_l_out, indices_split, axis_w)
        b_u_out_list = K.split(b_u_out, indices_split, self.axis)
        b_l_out_list = K.split(b_l_out, indices_split, self.axis)

        bounds = [[w_u_out_list[i], b_u_out_list[i], w_l_out_list[i], b_l_out_list[i]] for i in range(n_elem)]

        return bounds


class BackwardMultiply(BackwardMerge):
    """Backward  LiRPA of Multiply"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonMultiply):
            raise KeyError()

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        n_elem = len(inputs_list)

        if n_elem != 2:
            raise NotImplementedError("This layer is intended to merge only 2 layers.")

        return backward_multiply(
            inputs_list[0],
            inputs_list[1],
            w_u_out,
            b_u_out,
            w_l_out,
            b_l_out,
            perturbation_domain=self.layer.perturbation_domain,
            mode=self.mode,
            dc_decomp=self.dc_decomp,
        )


class BackwardDot(BackwardMerge):
    """Backward  LiRPA of Dot"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        if not isinstance(layer, DecomonDot):
            raise KeyError()

        self.axes = [i for i in self.layer.axes]
        self.op = BackwardAdd(self.layer)

        raise NotImplementedError()

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[List[keras.KerasTensor]]:
        w_u_out, b_u_out, w_l_out, b_l_out = get_identity_lirpa(inputs)

        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        n_elem = len(inputs_list)

        if n_elem != 2:
            raise NotImplementedError("This layer is intended to merge only 2 layers.")

        # permute dimensions and reshape
        inputs_0 = inputs_list[0]
        inputs_1 = inputs_list[1]

        n_0 = len(inputs_0[0].shape) - 2
        n_1 = len(inputs_1[0].shape) - 2

        input_0_permuted = permute_dimensions(inputs_0, self.axes[0], mode=self.mode, dc_decomp=self.dc_decomp)
        input_1_permuted = permute_dimensions(inputs_1, self.axes[1], mode=self.mode, dc_decomp=self.dc_decomp)

        inputs_0_broadcasted = broadcast(input_0_permuted, n_1, -1, mode=self.mode, dc_decomp=self.dc_decomp)
        inputs_1_broadcasted = broadcast(input_1_permuted, n_0, 2, mode=self.mode, dc_decomp=self.dc_decomp)

        inputs_multiplied = multiply(
            inputs_0_broadcasted,
            inputs_1_broadcasted,
            dc_decomp=self.layer.dc_decomp,
            perturbation_domain=self.perturbation_domain,
            mode=self.mode,
        )

        inputs_add = []
        for elem in split(inputs_multiplied, axis=self.axes[0], mode=self.mode):
            inputs_add += elem

        bounds = self.op.call(inputs_add)
        n = len(inputs_add)
        bounds = [[1.0 / n * elem for elem in bounds_i] for bounds_i in bounds]

        # concatenate
        shape = [-1, 1] + list(inputs_multiplied[0].shape[1:]) + list(w_u_out.shape[3:])

        bounds_reshaped = [[K.reshape(elem[0], shape), elem[1], K.reshape(elem[2], shape), elem[3]] for elem in bounds]
        w_u = K.concatenate([elem[0] for elem in bounds_reshaped], 1)
        b_u = sum([elem[1] for elem in bounds_reshaped], 1)
        w_l = K.concatenate([elem[2] for elem in bounds_reshaped], 1)
        b_l = sum([elem[3] for elem in bounds_reshaped])

        bounds_m_0, bounds_m_1 = backward_multiply(
            inputs_0_broadcasted,
            inputs_1_broadcasted,
            w_u,
            b_u,
            w_l,
            b_l,
            perturbation_domain=self.perturbation_domain,
            mode=self.mode,
            dc_decomp=self.dc_decomp,
        )

        shape_0 = [-1, 1] + list(input_0_permuted[0].shape[1:]) + list(w_u.shape[3:])
        shape_1 = [-1, 1] + list(input_1_permuted[0].shape[1:]) + list(w_u.shape[3:])

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
            K.transpose(bounds_m_0[0], index_0),
            bounds_m_0[1],
            K.transpose(bounds_m_0[2], index_0),
            bounds_m_0[3],
        ]
        bounds_m_1 = [
            K.transpose(bounds_m_1[0], index_1),
            bounds_m_1[1],
            K.transpose(bounds_m_1[2], index_1),
            bounds_m_1[3],
        ]

        return [bounds_m_0, bounds_m_1]
