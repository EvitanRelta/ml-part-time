from abc import ABC, abstractmethod
from typing import Iterator, Literal, overload

import torch
from torch import Tensor, nn
from typing_extensions import override

from ..preprocessing.solver_variables import (
    InputLayerVariables,
    IntermediateLayerVariables,
    LayerVariables,
    OutputLayerVariables,
    SolverVariables,
)
from .solver_utils import bracket_minus, bracket_plus


class SolverLayerList(nn.ModuleList):
    """Wrapper around `ModuleList` to contain `SolverLayer` modules."""

    def __init__(self, vars: SolverVariables):
        layers: list[SolverLayer] = [SolverInputLayer(vars.layer_vars[0])]
        for i in range(1, len(vars.layer_vars) - 1):
            layers.append(SolverIntermediateLayer(vars.layer_vars[i]))
        layers.append(SolverOutputLayer(vars.layer_vars[-1]))
        super().__init__(layers)

    def __iter__(self) -> Iterator["SolverLayer"]:
        return super().__iter__()  # type: ignore

    # fmt: off
    @overload
    def __getitem__(self, i: Literal[0]) -> "SolverInputLayer": ...
    @overload
    def __getitem__(self, i: Literal[-1]) -> "SolverOutputLayer": ...
    @overload
    def __getitem__(self, i: int) -> "SolverIntermediateLayer": ...
    # fmt: on
    def __getitem__(self, i: int) -> "SolverLayer":
        return super().__getitem__(i)  # type: ignore


class SolverLayer(ABC, nn.Module):
    """Abstract base class for all solver layers."""

    def __init__(self, vars: LayerVariables) -> None:
        super().__init__()
        self.vars = vars

    # fmt: off
    @abstractmethod
    def forward(self, V_next: Tensor) -> Tensor: ...

    @abstractmethod
    def clamp_parameters(self) -> None: ...

    @abstractmethod
    def get_obj_sum(self) -> Tensor: ...
    # fmt: on


class SolverInputLayer(SolverLayer):
    @override
    def __init__(self, vars: InputLayerVariables) -> None:
        super().__init__(vars)
        self.vars: InputLayerVariables

    # fmt: off
    @override
    def forward(self, V_next: Tensor) -> Tensor: raise NotImplementedError()

    @override
    def clamp_parameters(self) -> None: ...

    @override
    def get_obj_sum(self) -> Tensor: raise NotImplementedError()
    # fmt: on


class SolverIntermediateLayer(SolverLayer):
    @override
    def __init__(self, vars: IntermediateLayerVariables) -> None:
        super().__init__(vars)
        self.vars: IntermediateLayerVariables
        self.pi_i: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.P_i.size(0)))
        )
        self.alpha_i: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.num_unstable))
        )
        self.V_hat_i: Tensor | None = None

    @override
    def forward(self, V_next: Tensor) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        W_next, num_batches, num_neurons, num_unstable, P_i, P_hat_i, C_i, stably_act_mask, stably_deact_mask, unstable_mask, pi_i, alpha_i, U_i, L_i = self.vars.W_next, self.vars.num_batches, self.vars.num_neurons, self.vars.num_unstable, self.vars.P_i, self.vars.P_hat_i, self.vars.C_i, self.vars.stably_act_mask, self.vars.stably_deact_mask, self.vars.unstable_mask, self.pi_i, self.alpha_i, self.vars.U_i, self.vars.L_i  # fmt: skip
        device = V_next.device

        V_i: Tensor = torch.zeros((num_batches, num_neurons)).to(device)

        # Stably activated.
        stably_activated_V_i: Tensor = V_next @ W_next - C_i
        V_i[:, stably_act_mask] = stably_activated_V_i[:, stably_act_mask]

        # Stably deactivated.
        V_i[:, stably_deact_mask] = -C_i[:, stably_deact_mask]

        # Unstable.
        if num_unstable == 0:
            return V_i

        V_hat_i = (V_next @ W_next)[:, unstable_mask] - pi_i @ P_hat_i
        self.V_hat_i = V_hat_i

        V_i[:, unstable_mask] = (
            (bracket_plus(V_hat_i) * U_i[unstable_mask]) / (U_i[unstable_mask] - L_i[unstable_mask])
            - C_i[:, unstable_mask]
            - alpha_i * bracket_minus(V_hat_i)
            - pi_i @ P_i
        )
        return V_i

    @override
    def clamp_parameters(self) -> None:
        self.pi_i.clamp_(min=0)
        self.alpha_i.clamp_(min=0, max=1)

    @override
    def get_obj_sum(self) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        num_batches, L_i, U_i, unstable_mask, p_i, pi_i = self.vars.num_batches, self.vars.L_i, self.vars.U_i, self.vars.unstable_mask, self.vars.p_i, self.pi_i  # fmt: skip
        device = self.pi_i.device

        if self.vars.num_unstable == 0:
            return torch.zeros((num_batches,)).to(device)

        assert self.V_hat_i is not None
        V_hat_i = self.V_hat_i
        return (
            torch.sum(
                (bracket_plus(V_hat_i) * U_i[unstable_mask] * L_i[unstable_mask])
                / (U_i[unstable_mask] - L_i[unstable_mask]),
                dim=1,
            )
            - pi_i @ p_i
        )


class SolverOutputLayer(SolverLayer):
    @override
    def __init__(self, vars: OutputLayerVariables) -> None:
        super().__init__(vars)
        self.vars: OutputLayerVariables
        self.gamma: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.H.size(0)))
        )

    @override
    def forward(self, V_next: Tensor = torch.empty(0)) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        H, gamma = self.vars.H, self.gamma  # fmt: skip

        V_L = (-H.T @ gamma.T).T
        assert V_L.dim() == 2
        return V_L

    @override
    def clamp_parameters(self) -> None:
        self.gamma.clamp_(min=0)

    @override
    def get_obj_sum(self) -> Tensor:
        device = self.gamma.device
        return torch.zeros((1,)).to(device)
