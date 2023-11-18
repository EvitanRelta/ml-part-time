from typing import Iterator, List, Literal, overload

from torch import Tensor, nn

from . import preprocessing_utils
from .solver_inputs import SolverInputs


class SolverVariables(nn.Module):
    """Contains all variables used during solving."""

    def __init__(self, inputs: SolverInputs) -> None:
        super().__init__()
        self.d: Tensor
        self.register_buffer("d", inputs.d)

        preprocessing_utils.freeze_model(inputs.model)
        W, b = preprocessing_utils.decompose_model(inputs.model)
        (
            stably_act_masks,
            stably_deact_masks,
            unstable_masks,
        ) = preprocessing_utils.get_masks(inputs.L, inputs.U)

        # Initially set to solve for input layer.
        C, self.solve_coords = preprocessing_utils.get_C_for_layer(0, unstable_masks)

        self.layer_vars: LayerVariablesList = SolverVariables._split_vars_per_layer(
            inputs, W, b, stably_act_masks, stably_deact_masks, unstable_masks, C
        )

    def solve_for_layer(self, layer_index: int) -> None:
        C, self.solve_coords = preprocessing_utils.get_C_for_layer(layer_index, self.unstable_masks)
        for i in range(len(self.layer_vars)):
            self.layer_vars[i].set_C_i(C[i])

    @property
    def L(self) -> List[Tensor]:
        return [x.L_i for x in self.layer_vars]

    @property
    def U(self) -> List[Tensor]:
        return [x.U_i for x in self.layer_vars]

    @property
    def H(self) -> Tensor:
        return self.layer_vars[-1].H

    @property
    def W(self) -> List[Tensor]:
        return [self.layer_vars[i].W_i for i in range(1, len(self.layer_vars))]

    @property
    def b(self) -> List[Tensor]:
        return [self.layer_vars[i].b_i for i in range(1, len(self.layer_vars))]

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
    def C(self) -> List[Tensor]:
        return [x.C_i for x in self.layer_vars]

    @staticmethod
    def _split_vars_per_layer(
        inputs: SolverInputs,
        W: List[Tensor],
        b: List[Tensor],
        stably_act_masks: List[Tensor],
        stably_deact_masks: List[Tensor],
        unstable_masks: List[Tensor],
        C: List[Tensor],
    ) -> "LayerVariablesList":
        layer_var_list: List[LayerVariables] = []

        # First-layer inputs.
        layer_var_list.append(
            InputLayerVariables(
                L_i=inputs.L[0],
                U_i=inputs.U[0],
                stably_act_mask=stably_act_masks[0],
                stably_deact_mask=stably_deact_masks[0],
                unstable_mask=unstable_masks[0],
                C_i=C[0],
            )
        )

        # Intermediate-layer inputs.
        for i in range(1, len(W)):
            layer_var_list.append(
                IntermediateLayerVariables(
                    L_i=inputs.L[i],
                    U_i=inputs.U[i],
                    stably_act_mask=stably_act_masks[i],
                    stably_deact_mask=stably_deact_masks[i],
                    unstable_mask=unstable_masks[i],
                    C_i=C[i],
                    W_i=W[i - 1],
                    b_i=b[i - 1],
                    W_next=W[i],
                    P_i=inputs.P[i - 1],
                    P_hat_i=inputs.P_hat[i - 1],
                    p_i=inputs.p[i - 1],
                )
            )

        # Last-layer inputs.
        layer_var_list.append(
            OutputLayerVariables(
                L_i=inputs.L[-1],
                U_i=inputs.U[-1],
                stably_act_mask=stably_act_masks[-1],
                stably_deact_mask=stably_deact_masks[-1],
                unstable_mask=unstable_masks[-1],
                C_i=C[-1],
                W_i=W[-1],
                b_i=b[-1],
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
        L_i: Tensor,
        U_i: Tensor,
        stably_act_mask: Tensor,
        stably_deact_mask: Tensor,
        unstable_mask: Tensor,
        C_i: Tensor,
    ) -> None:
        super().__init__()
        self.L_i: Tensor
        self.U_i: Tensor
        self.stably_act_mask: Tensor
        self.stably_deact_mask: Tensor
        self.unstable_mask: Tensor
        self.C_i: Tensor

        self.register_buffer("L_i", L_i)
        self.register_buffer("U_i", U_i)
        self.register_buffer("stably_act_mask", stably_act_mask)
        self.register_buffer("stably_deact_mask", stably_deact_mask)
        self.register_buffer("unstable_mask", unstable_mask)
        self.register_buffer("C_i", C_i)

    def set_C_i(self, C_i: Tensor) -> None:
        self.register_buffer("C_i", C_i)

    @property
    def num_batches(self) -> int:
        return self.C_i.size(0)

    @property
    def num_neurons(self) -> int:
        return len(self.L_i)

    @property
    def num_unstable(self) -> int:
        return int(self.unstable_mask.sum().item())


class InputLayerVariables(LayerVariables):
    ...


class IntermediateLayerVariables(LayerVariables):
    def __init__(
        self,
        W_i: Tensor,
        b_i: Tensor,
        W_next: Tensor,
        P_i: Tensor,
        P_hat_i: Tensor,
        p_i: Tensor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.W_i: Tensor
        self.b_i: Tensor
        self.W_next: Tensor
        self.P_i: Tensor
        self.P_hat_i: Tensor
        self.p_i: Tensor

        self.register_buffer("W_i", W_i)
        self.register_buffer("b_i", b_i)
        self.register_buffer("W_next", W_next)
        self.register_buffer("P_i", P_i)
        self.register_buffer("P_hat_i", P_hat_i)
        self.register_buffer("p_i", p_i)


class OutputLayerVariables(LayerVariables):
    def __init__(self, W_i: Tensor, b_i: Tensor, H: Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.W_i: Tensor
        self.b_i: Tensor
        self.H: Tensor

        self.register_buffer("W_i", W_i)
        self.register_buffer("b_i", b_i)
        self.register_buffer("H", H)
