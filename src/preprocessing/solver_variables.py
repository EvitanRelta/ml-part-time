from typing import Iterator, List, Literal, overload

from torch import Tensor, nn

from . import preprocessing_utils
from .solver_inputs import SolverInputs
from .transpose import UnaryForwardModule, transpose_model


class SolverVariables(nn.Module):
    """Contains all variables used during solving."""

    def __init__(self, inputs: SolverInputs) -> None:
        super().__init__()
        self.d: Tensor
        self.register_buffer("d", inputs.d)

        preprocessing_utils.freeze_model(inputs.model)
        transposed_layers, b_list = transpose_model(inputs.model)
        (
            stably_act_masks,
            stably_deact_masks,
            unstable_masks,
        ) = preprocessing_utils.get_masks(inputs.L_list, inputs.U_list)

        # Initially set to solve for input layer.
        C_list, self.solve_coords = preprocessing_utils.get_C_for_layer(0, unstable_masks)

        self.layer_vars: LayerVariablesList = SolverVariables._split_vars_per_layer(
            inputs,
            transposed_layers,
            b_list,
            stably_act_masks,
            stably_deact_masks,
            unstable_masks,
            C_list,
        )

    def solve_for_layer(self, layer_index: int) -> None:
        C_list, self.solve_coords = preprocessing_utils.get_C_for_layer(
            layer_index, self.unstable_masks
        )
        for i in range(len(self.layer_vars)):
            self.layer_vars[i].set_C(C_list[i])

    @property
    def L_list(self) -> List[Tensor]:
        return [x.L for x in self.layer_vars]

    @property
    def U_list(self) -> List[Tensor]:
        return [x.U for x in self.layer_vars]

    @property
    def H(self) -> Tensor:
        return self.layer_vars[-1].H

    @property
    def b_list(self) -> List[Tensor]:
        return [self.layer_vars[i].b for i in range(1, len(self.layer_vars))]

    @property
    def stably_act_masks(self) -> List[Tensor]:
        return [x.stably_act_mask for x in self.layer_vars]

    @property
    def stably_deact_masks(self) -> List[Tensor]:
        return [x.stably_deact_mask for x in self.layer_vars]

    @property
    def unstable_masks(self) -> List[Tensor]:
        return [x.unstable_mask for x in self.layer_vars]

    @property
    def C_list(self) -> List[Tensor]:
        return [x.C for x in self.layer_vars]

    @staticmethod
    def _split_vars_per_layer(
        inputs: SolverInputs,
        transposed_layers: List[UnaryForwardModule],
        b_list: List[Tensor],
        stably_act_masks: List[Tensor],
        stably_deact_masks: List[Tensor],
        unstable_masks: List[Tensor],
        C_list: List[Tensor],
    ) -> "LayerVariablesList":
        layer_var_list: List[LayerVariables] = []

        # First-layer inputs.
        layer_var_list.append(
            InputLayerVariables(
                L=inputs.L_list[0],
                U=inputs.U_list[0],
                stably_act_mask=stably_act_masks[0],
                stably_deact_mask=stably_deact_masks[0],
                unstable_mask=unstable_masks[0],
                C=C_list[0],
            )
        )

        # Intermediate-layer inputs.
        for i in range(1, len(b_list)):
            layer_var_list.append(
                IntermediateLayerVariables(
                    L=inputs.L_list[i],
                    U=inputs.U_list[i],
                    stably_act_mask=stably_act_masks[i],
                    stably_deact_mask=stably_deact_masks[i],
                    unstable_mask=unstable_masks[i],
                    C=C_list[i],
                    transposed_layer=transposed_layers[i - 1],
                    b=b_list[i - 1],
                    transposed_layer_next=transposed_layers[i],
                    P=inputs.P_list[i - 1],
                    P_hat=inputs.P_hat_list[i - 1],
                    p=inputs.p_list[i - 1],
                )
            )

        # Last-layer inputs.
        layer_var_list.append(
            OutputLayerVariables(
                L=inputs.L_list[-1],
                U=inputs.U_list[-1],
                stably_act_mask=stably_act_masks[-1],
                stably_deact_mask=stably_deact_masks[-1],
                unstable_mask=unstable_masks[-1],
                C=C_list[-1],
                transposed_layer=transposed_layers[-1],
                b=b_list[-1],
                H=inputs.H,
            )
        )

        return LayerVariablesList(layer_var_list)


# ==============================================================================
#                     Classes for storing variables per layer
# ==============================================================================
class LayerVariablesList(nn.ModuleList):
    """Wrapper around `ModuleList` to contain `LayerVariables` modules."""

    def __init__(self, layer_vars: List["LayerVariables"]) -> None:
        assert isinstance(layer_vars[0], InputLayerVariables)
        assert isinstance(layer_vars[-1], OutputLayerVariables)
        for i in range(1, len(layer_vars) - 1):
            assert isinstance(layer_vars[i], IntermediateLayerVariables)
        super().__init__(layer_vars)

    def __iter__(self) -> Iterator["LayerVariables"]:
        return super().__iter__()  # type: ignore

    # fmt: off
    @overload
    def __getitem__(self, i: Literal[0]) -> "InputLayerVariables": ...
    @overload
    def __getitem__(self, i: Literal[-1]) -> "OutputLayerVariables": ...
    @overload
    def __getitem__(self, i: int) -> "IntermediateLayerVariables": ...
    # fmt: on
    def __getitem__(self, i: int) -> "LayerVariables":
        return super().__getitem__(i)  # type: ignore


class LayerVariables(nn.Module):
    def __init__(
        self,
        L: Tensor,
        U: Tensor,
        stably_act_mask: Tensor,
        stably_deact_mask: Tensor,
        unstable_mask: Tensor,
        C: Tensor,
    ) -> None:
        super().__init__()
        self.L: Tensor
        self.U: Tensor
        self.stably_act_mask: Tensor
        self.stably_deact_mask: Tensor
        self.unstable_mask: Tensor
        self.C: Tensor

        self.register_buffer("L", L)
        self.register_buffer("U", U)
        self.register_buffer("stably_act_mask", stably_act_mask)
        self.register_buffer("stably_deact_mask", stably_deact_mask)
        self.register_buffer("unstable_mask", unstable_mask)
        self.register_buffer("C", C)

    def set_C(self, C: Tensor) -> None:
        self.register_buffer("C", C)

    @property
    def num_batches(self) -> int:
        return self.C.size(0)

    @property
    def num_neurons(self) -> int:
        return len(self.L)

    @property
    def num_unstable(self) -> int:
        return int(self.unstable_mask.sum().item())


class InputLayerVariables(LayerVariables):
    ...


class IntermediateLayerVariables(LayerVariables):
    def __init__(
        self,
        transposed_layer: UnaryForwardModule,
        b: Tensor,
        transposed_layer_next: UnaryForwardModule,
        P: Tensor,
        P_hat: Tensor,
        p: Tensor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transposed_layer = transposed_layer
        self.transposed_layer_next = transposed_layer_next

        self.b: Tensor
        self.P: Tensor
        self.P_hat: Tensor
        self.p: Tensor

        self.register_buffer("b", b)
        self.register_buffer("P", P)
        self.register_buffer("P_hat", P_hat)
        self.register_buffer("p", p)


class OutputLayerVariables(LayerVariables):
    def __init__(
        self,
        transposed_layer: UnaryForwardModule,
        b: Tensor,
        H: Tensor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transposed_layer = transposed_layer

        self.b: Tensor
        self.H: Tensor

        self.register_buffer("b", b)
        self.register_buffer("H", H)
