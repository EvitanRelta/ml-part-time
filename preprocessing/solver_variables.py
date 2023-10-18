from typing import Iterator, Literal, overload

from torch import Tensor, nn

from . import preprocessing_utils
from .solver_inputs import SolverInputs


class SolverVariables(nn.Module):
    """Contains all variables used during solving."""

    def __init__(self, inputs: SolverInputs) -> None:
        super().__init__()
        self.inputs = inputs
        self.d = inputs.d

        preprocessing_utils.freeze_model(inputs.model)
        self.num_layers, self.W, self.b = preprocessing_utils.decompose_model(inputs.model)
        (
            self.stably_act_masks,
            self.stably_deact_masks,
            self.unstable_masks,
        ) = preprocessing_utils.get_masks(inputs.L, inputs.U)

        # Initially set to solve for input layer.
        self.C = preprocessing_utils.get_C_for_layer(0, self.unstable_masks)
        self.layer_vars: LayerVariablesList = self._split_vars_per_layer()

    def solve_for_layer(self, layer_index: int) -> None:
        self.C = preprocessing_utils.get_C_for_layer(layer_index, self.unstable_masks)
        for i in range(len(self.layer_vars)):
            self.layer_vars[i].set_C_i(self.C[i])

    def _split_vars_per_layer(self) -> "LayerVariablesList":
        layer_var_list: list[LayerVariables] = []

        # First-layer inputs.
        layer_var_list.append(
            InputLayerVariables(
                L_i=self.inputs.L[0],
                U_i=self.inputs.U[0],
                stably_act_mask=self.stably_act_masks[0],
                stably_deact_mask=self.stably_deact_masks[0],
                unstable_mask=self.unstable_masks[0],
                C_i=self.C[0],
            )
        )

        # Intermediate-layer inputs.
        for i in range(1, self.num_layers):
            layer_var_list.append(
                IntermediateLayerVariables(
                    L_i=self.inputs.L[i],
                    U_i=self.inputs.U[i],
                    stably_act_mask=self.stably_act_masks[i],
                    stably_deact_mask=self.stably_deact_masks[i],
                    unstable_mask=self.unstable_masks[i],
                    C_i=self.C[i],
                    W_i=self.W[i],
                    b_i=self.b[i],
                    W_next=self.W[i + 1],
                    P_i=self.inputs.P[i - 1],
                    P_hat_i=self.inputs.P_hat[i - 1],
                    p_i=self.inputs.p[i - 1],
                )
            )

        # Last-layer inputs.
        layer_var_list.append(
            OutputLayerVariables(
                L_i=self.inputs.L[-1],
                U_i=self.inputs.U[-1],
                stably_act_mask=self.stably_act_masks[-1],
                stably_deact_mask=self.stably_deact_masks[-1],
                unstable_mask=self.unstable_masks[-1],
                C_i=self.C[-1],
                W_i=self.W[-1],
                b_i=self.b[-1],
                H=self.inputs.H,
            )
        )

        return LayerVariablesList(layer_var_list)


# ==============================================================================
#                     Classes for storing variables per layer
# ==============================================================================
class LayerVariablesList(nn.ModuleList):
    """Wrapper around `ModuleList` to contain `LayerVariables` modules."""

    def __init__(self, layer_vars: list["LayerVariables"]) -> None:
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
        self.L_i = L_i
        self.U_i = U_i
        self.stably_act_mask = stably_act_mask
        self.stably_deact_mask = stably_deact_mask
        self.unstable_mask = unstable_mask
        self.C_i = C_i

    def set_C_i(self, C_i: Tensor) -> None:
        self.C_i = C_i

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
        self.W_i = W_i
        self.b_i = b_i
        self.W_next = W_next
        self.P_i = P_i
        self.P_hat_i = P_hat_i
        self.p_i = p_i


class OutputLayerVariables(LayerVariables):
    def __init__(self, W_i: Tensor, b_i: Tensor, H: Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.W_i = W_i
        self.b_i = b_i
        self.H = H
