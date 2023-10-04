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
        self.inputs = inputs

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
        self.inputs: InputLayerInputs

    def forward(self, V_next: Tensor) -> Tensor:
        raise NotImplementedError()

    def clamp_parameters(self) -> None:
        pass

    def get_obj_sum(self) -> Tensor:
        raise NotImplementedError()


class SolverOutputLayer(SolverLayer):
    def __init__(self, inputs: OutputLayerInputs) -> None:
        super().__init__(inputs)
        self.inputs: OutputLayerInputs

        if self.inputs.initial_gamma is not None:
            self.gamma: nn.Parameter = nn.Parameter(
                self.inputs.initial_gamma
                if self.inputs.initial_gamma.dim() == 2
                else torch.stack(
                    [self.inputs.initial_gamma.clone().detach()] * self.inputs.batches
                )  # Copy initial values for all batches.
            )
        else:
            self.gamma: nn.Parameter = nn.Parameter(
                torch.rand((self.inputs.batches, self.inputs.H.size(0)))
            )
        assert self.gamma.shape == (
            self.inputs.batches,
            self.inputs.H.size(0),
        )

    @override
    def forward(self, V_next: Tensor = torch.empty(0)) -> Tensor:
        H, gamma = self.inputs.H, self.gamma
        V_L = (-H.T @ gamma.T).T
        assert V_L.dim() == 2
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
        self.inputs: IntermediateLayerInputs

        assert self.inputs.P_i.size(1) == self.inputs.P_hat_i.size(1) == self.inputs.num_unstable

        if self.inputs.initial_pi_i is not None:
            self.pi_i: nn.Parameter = nn.Parameter(
                self.inputs.initial_pi_i
                if self.inputs.initial_pi_i.dim() == 2
                else torch.stack(
                    [self.inputs.initial_pi_i.clone().detach()] * self.inputs.batches
                )  # Copy initial values for all batches.
            )
        else:
            self.pi_i: nn.Parameter = nn.Parameter(
                torch.rand((self.inputs.batches, self.inputs.P_i.size(0)))
            )
        assert self.pi_i.shape == (
            self.inputs.batches,
            self.inputs.P_i.size(0),
        )

        if self.inputs.initial_alpha_i is not None:
            self.alpha_i: nn.Parameter = nn.Parameter(
                self.inputs.initial_alpha_i
                if self.inputs.initial_alpha_i.dim() == 2
                else torch.stack(
                    [self.inputs.initial_alpha_i.clone().detach()] * self.inputs.batches
                )  # Copy initial values for all batches.
            )
        else:
            self.alpha_i: nn.Parameter = nn.Parameter(
                torch.rand((self.inputs.batches, self.inputs.num_unstable))
            )
        assert self.alpha_i.shape == (
            self.inputs.batches,
            self.inputs.num_unstable,
        )

        self.V_hat_i: Tensor | None = None

    @override
    def forward(self, V_next: Tensor) -> Tensor:
        # fmt: off
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        W_next, batches, num_neurons, num_unstable, P_i, P_hat_i, C_i, stably_act_mask, stably_deact_mask, unstable_mask, pi_i, alpha_i, U_i, L_i = self.inputs.W_next, self.inputs.batches, self.inputs.num_neurons, self.inputs.num_unstable, self.inputs.P_i, self.inputs.P_hat_i, self.inputs.C_i, self.inputs.stably_act_mask, self.inputs.stably_deact_mask, self.inputs.unstable_mask, self.pi_i, self.alpha_i, self.inputs.U_i, self.inputs.L_i
        # fmt: on

        V_i: Tensor = torch.zeros((batches, num_neurons))

        # Stably activated.
        stably_activated_V_i: Tensor = V_next @ W_next - C_i
        V_i[:, stably_act_mask] = stably_activated_V_i[:, stably_act_mask]
        # torch.tensor(
        #     [V_next.T @ W_next[j] - C_i[j] for j in range(num_neurons)]
        # )

        # Stably deactivated.
        V_i[:, stably_deact_mask] = -C_i[:, stably_deact_mask]

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
        # fmt: off
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        batches, L_i, U_i, unstable_mask, p_i, pi_i = self.inputs.batches, self.inputs.L_i, self.inputs.U_i, self.inputs.unstable_mask, self.inputs.p_i, self.pi_i
        # fmt: on

        if self.inputs.num_unstable == 0:
            return torch.zeros((batches,))

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
