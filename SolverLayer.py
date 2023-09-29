from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from typing_extensions import override

from inputs_dataclasses import (
    InputLayerInputs,
    IntermediateLayerInputs,
    LayerInputs,
    OutputLayerInputs,
)
from utils import bracket_minus, bracket_plus


class SolverLayer(ABC, nn.Module):
    def __init__(self, inputs: LayerInputs) -> None:
        super().__init__()
        self.num_neurons = inputs.num_neurons
        self.L_i = inputs.L_i
        self.U_i = inputs.U_i
        self.C_i: Tensor = torch.zeros((self.num_neurons,))

    def clear_C(self) -> None:
        self.C_i: Tensor = torch.zeros((self.num_neurons,))

    def set_C(self, j: int, is_min: bool) -> None:
        self.C_i[j] = 1 if is_min else -1

    @abstractmethod
    def forward(self, V_next: Tensor) -> Tensor:
        ...

    @abstractmethod
    def clamp_parameters(self) -> None:
        ...

    @abstractmethod
    def get_obj_sum(self) -> Tensor:
        ...


class SolverInputLayer(SolverLayer):
    def __init__(self, inputs: InputLayerInputs) -> None:
        super().__init__(inputs)

    def forward(self, V_next: Tensor) -> Tensor:
        raise NotImplementedError()

    def clamp_parameters(self) -> None:
        pass

    def get_obj_sum(self) -> Tensor:
        raise NotImplementedError()


class SolverOutputLayer(SolverLayer):
    def __init__(self, inputs: OutputLayerInputs) -> None:
        super().__init__(inputs)
        self.W_i = inputs.W_i
        self.b_i = inputs.b_i
        self.H = inputs.H

        if inputs.initial_gamma is not None:
            self.gamma: nn.Parameter = nn.Parameter(inputs.initial_gamma)
        else:
            self.gamma: nn.Parameter = nn.Parameter(torch.rand((self.H.size(0),)))
        assert self.gamma.shape == (self.H.size(0),)

    @override
    def forward(self, V_next: Tensor = torch.empty(0)) -> Tensor:
        H, gamma = self.H, self.gamma
        V_L = (-H.T @ gamma).squeeze()
        assert V_L.dim() == 1
        return V_L

    @override
    def clamp_parameters(self) -> None:
        self.gamma.clamp_(min=0)

    @override
    def get_obj_sum(self) -> Tensor:
        return torch.zeros((1,))


class SolverIntermediateLayer(SolverLayer):
    def __init__(self, inputs: IntermediateLayerInputs) -> None:
        super().__init__(inputs)
        self.W_i = inputs.W_i
        self.b_i = inputs.b_i
        self.W_next = inputs.W_next
        self.P_i = inputs.P_i
        self.P_hat_i = inputs.P_hat_i
        self.p_i = inputs.p_i

        self.stably_act_mask: Tensor = self.L_i >= 0
        self.stably_deact_mask: Tensor = self.U_i <= 0
        self.unstable_mask: Tensor = (self.L_i < 0) & (self.U_i > 0)
        assert torch.all((self.stably_act_mask + self.stably_deact_mask + self.unstable_mask) == 1)

        self.num_unstable: int = int(self.unstable_mask.sum().item())

        assert self.P_i.size(1) == self.P_hat_i.size(1) == self.num_unstable

        if inputs.initial_pi_i is not None:
            self.pi_i: nn.Parameter = nn.Parameter(inputs.initial_pi_i)
        else:
            self.pi_i: nn.Parameter = nn.Parameter(torch.rand((self.P_i.size(0),)))
        assert self.pi_i.shape == (self.P_i.size(0),)

        if inputs.initial_alpha_i is not None:
            self.alpha_i: nn.Parameter = nn.Parameter(inputs.initial_alpha_i)
        else:
            self.alpha_i: nn.Parameter = nn.Parameter(torch.rand((self.num_unstable,)))
        assert self.alpha_i.shape == (self.num_unstable,)

        self.V_hat_i: Tensor | None = None

    @override
    def forward(self, V_next: Tensor) -> Tensor:
        # fmt: off
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        W_next, num_neurons, num_unstable, P_i, P_hat_i, C_i, stably_act_mask, stably_deact_mask, unstable_mask, pi_i, alpha_i, U_i, L_i = self.W_next, self.num_neurons, self.num_unstable, self.P_i, self.P_hat_i, self.C_i, self.stably_act_mask, self.stably_deact_mask, self.unstable_mask, self.pi_i, self.alpha_i, self.U_i, self.L_i
        # fmt: on

        V_i: Tensor = torch.zeros((num_neurons,))

        # Stably activated.
        stably_activated_V_i: Tensor = (V_next @ W_next.T).squeeze() - C_i
        V_i[stably_act_mask] = stably_activated_V_i[stably_act_mask]
        # torch.tensor(
        #     [V_next.T @ W_next[j] - C_i[j] for j in range(num_neurons)]
        # )

        # Stably deactivated.
        V_i[stably_deact_mask] = -C_i[stably_deact_mask]

        if num_unstable == 0:
            return V_i

        V_hat_i = (V_next @ W_next.T).squeeze()[unstable_mask] - pi_i @ P_hat_i
        self.V_hat_i = V_hat_i

        V_i[unstable_mask] = (
            (bracket_plus(V_hat_i) * U_i[unstable_mask]) / (U_i[unstable_mask] - L_i[unstable_mask])
            - C_i[unstable_mask]
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
        # fmt: off
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        L_i, U_i, unstable_mask, p_i, pi_i = self.L_i, self.U_i, self.unstable_mask, self.p_i, self.pi_i
        # fmt: on

        if self.num_unstable == 0:
            return torch.zeros((1,))

        assert self.V_hat_i is not None
        V_hat_i = self.V_hat_i
        return (
            torch.sum(
                (bracket_plus(V_hat_i) * U_i[unstable_mask] * L_i[unstable_mask])
                / (U_i[unstable_mask] - L_i[unstable_mask])
            )
            - pi_i @ p_i
        )
