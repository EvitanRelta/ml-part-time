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
        self.pi_i: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.P_i.size(0)))
        )
        self.alpha_i: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.num_unstable))
        )
        self.V_hat_i: Optional[Tensor] = None

    @override
    def forward(self, V_next: Tensor) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        W_next, num_batches, num_neurons, num_unstable, P_i, P_hat_i, C_i, stably_act_mask, stably_deact_mask, unstable_mask, pi_i, alpha_i, U_i, L_i = self.vars.W_next, self.vars.num_batches, self.vars.num_neurons, self.vars.num_unstable, self.vars.P_i, self.vars.P_hat_i, self.vars.C_i, self.vars.stably_act_mask, self.vars.stably_deact_mask, self.vars.unstable_mask, self.pi_i, self.alpha_i, self.vars.U_i, self.vars.L_i  # fmt: skip
        device = V_next.device

        V_i: Tensor = torch.zeros((num_batches, num_neurons)).to(device)
        V_next_W_next = V_next @ W_next

        # Stably activated.
        stably_activated_V_i: Tensor = V_next_W_next - C_i
        V_i[:, stably_act_mask] = stably_activated_V_i[:, stably_act_mask]

        # Stably deactivated.
        V_i[:, stably_deact_mask] = -C_i[:, stably_deact_mask]

        # Unstable.
        if num_unstable == 0:
            return V_i

        V_hat_i = V_next_W_next[:, unstable_mask] - pi_i @ P_hat_i
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
