from typing import Optional

import torch
from torch import Tensor, nn
from typing_extensions import override

from ...preprocessing.solver_variables import IntermediateLayerVariables
from ..solver_utils import bracket_minus, bracket_plus
from .base_class import SolverLayer


class SolverIntermediate(SolverLayer):
    @override
    def __init__(self, vars: IntermediateLayerVariables) -> None:
        super().__init__(vars)
        self.vars: IntermediateLayerVariables
        self.pi: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.P.size(0)))
        )
        self.alpha: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.num_unstable))
        )
        self.V_hat: Optional[Tensor] = None

    @override
    def forward(self, V_next: Tensor) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        transposed_layer_next, num_batches, num_neurons, num_unstable, P, P_hat, C, stably_act_mask, stably_deact_mask, unstable_mask, pi, alpha, U, L = self.vars.transposed_layer_next, self.vars.num_batches, self.vars.num_neurons, self.vars.num_unstable, self.vars.P, self.vars.P_hat, self.vars.C, self.vars.stably_act_mask, self.vars.stably_deact_mask, self.vars.unstable_mask, self.pi, self.alpha, self.vars.U, self.vars.L  # fmt: skip
        device = V_next.device

        V: Tensor = torch.zeros((num_batches, num_neurons)).to(device)
        V_next_W_next = transposed_layer_next.forward(V_next)

        # Stably activated.
        stably_activated_V: Tensor = V_next_W_next - C
        V[:, stably_act_mask] = stably_activated_V[:, stably_act_mask]

        # Stably deactivated.
        V[:, stably_deact_mask] = -C[:, stably_deact_mask]

        # Unstable.
        if num_unstable == 0:
            return V

        V_hat = V_next_W_next[:, unstable_mask] - pi @ P_hat
        self.V_hat = V_hat

        V[:, unstable_mask] = (
            (bracket_plus(V_hat) * U[unstable_mask]) / (U[unstable_mask] - L[unstable_mask])
            - C[:, unstable_mask]
            - alpha * bracket_minus(V_hat)
            - pi @ P
        )
        return V

    @override
    def clamp_parameters(self) -> None:
        self.pi.clamp_(min=0)
        self.alpha.clamp_(min=0, max=1)

    @override
    def get_obj_sum(self) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        num_batches, L, U, unstable_mask, p, pi = self.vars.num_batches, self.vars.L, self.vars.U, self.vars.unstable_mask, self.vars.p, self.pi  # fmt: skip
        device = self.pi.device

        if self.vars.num_unstable == 0:
            return torch.zeros((num_batches,)).to(device)

        assert self.V_hat is not None
        V_hat = self.V_hat
        return (
            torch.sum(
                (bracket_plus(V_hat) * U[unstable_mask] * L[unstable_mask])
                / (U[unstable_mask] - L[unstable_mask]),
                dim=1,
            )
            - pi @ p
        )
