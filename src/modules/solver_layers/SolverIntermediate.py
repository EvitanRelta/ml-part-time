from typing import Optional

import torch
from torch import Tensor, nn
from typing_extensions import override

from ...preprocessing.transpose import UnaryForward
from ..solver_utils import bracket_minus, bracket_plus
from .base_class import SolverLayer


class SolverIntermediate(SolverLayer):
    @override
    def __init__(
        self,
        L: Tensor,
        U: Tensor,
        stably_act_mask: Tensor,
        stably_deact_mask: Tensor,
        unstable_mask: Tensor,
        C: Tensor,
        transposed_layer: UnaryForward,
        b: Tensor,
        transposed_layer_next: UnaryForward,
        P: Tensor,
        P_hat: Tensor,
        p: Tensor,
    ) -> None:
        super().__init__(L, U, stably_act_mask, stably_deact_mask, unstable_mask, C)
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

        self.pi: nn.Parameter = nn.Parameter(torch.rand((self.num_batches, self.P.size(0))))
        self.alpha: nn.Parameter = nn.Parameter(torch.rand((self.num_batches, self.num_unstable)))
        self.V_hat: Optional[Tensor] = None

    @override
    def set_C_and_reset(self, C: Tensor) -> None:
        super().set_C_and_reset(C)
        self.pi: nn.Parameter = nn.Parameter(torch.rand((self.num_batches, self.P.size(0))))
        self.alpha: nn.Parameter = nn.Parameter(torch.rand((self.num_batches, self.num_unstable)))
        self.V_hat: Optional[Tensor] = None

    @override
    def forward(self, V_next: Tensor) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        transposed_layer_next, num_batches, num_neurons, num_unstable, P, P_hat, C, stably_act_mask, stably_deact_mask, unstable_mask, pi, alpha, U, L = self.transposed_layer_next, self.num_batches, self.num_neurons, self.num_unstable, self.P, self.P_hat, self.C, self.stably_act_mask, self.stably_deact_mask, self.unstable_mask, self.pi, self.alpha, self.U, self.L  # fmt: skip
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
        num_batches, L, U, unstable_mask, p, pi = self.num_batches, self.L, self.U, self.unstable_mask, self.p, self.pi  # fmt: skip
        device = self.pi.device

        if self.num_unstable == 0:
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
