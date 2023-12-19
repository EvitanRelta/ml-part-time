from typing import Tuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from ...preprocessing.class_definitions import Bias
from ...preprocessing.transpose import UnaryForward
from ..solver_utils import bracket_minus, bracket_plus
from .base_class import SolverLayer


class IntermediateLayer(SolverLayer):
    """The solver layer for the model's "intermediate layers". These layers are
    evaluated after the solver-output-layer, as the computation propagates
    from output-layer to intermediate-layers to input-layer.
    """

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
        bias_module: Bias,
        transposed_layer_next: UnaryForward,
        P: Tensor,
        P_hat: Tensor,
        p: Tensor,
    ) -> None:
        super().__init__(L, U, stably_act_mask, stably_deact_mask, unstable_mask, C)
        self.transposed_layer = transposed_layer
        self.transposed_layer_next = transposed_layer_next
        self.bias_module = bias_module

        self.P: Tensor
        self.P_hat: Tensor
        self.p: Tensor
        self.register_buffer("P", P)
        self.register_buffer("P_hat", P_hat)
        self.register_buffer("p", p)

    @override
    def set_C_and_reset_parameters(self, C: Tensor) -> None:
        super().set_C_and_reset_parameters(C)
        self.pi: nn.Parameter = nn.Parameter(torch.rand((self.num_batches, self.P.size(0))).to(C))
        self.alpha: nn.Parameter = nn.Parameter(
            torch.rand((self.num_batches, self.num_unstable)).to(C)
        )

    def forward(self, V_next: Tensor, accum_sum: Tensor) -> Tuple[Tensor, Tensor]:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        bias_module, transposed_layer_next, num_batches, num_neurons, num_unstable, P, P_hat, p, C, stably_act_mask, stably_deact_mask, unstable_mask, pi, alpha, U, L = self.bias_module, self.transposed_layer_next, self.num_batches, self.num_neurons, self.num_unstable, self.P, self.P_hat, self.p, self.C, self.stably_act_mask, self.stably_deact_mask, self.unstable_mask, self.pi, self.alpha, self.U, self.L  # fmt: skip
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
            return V, accum_sum

        V_hat = V_next_W_next[:, unstable_mask] - pi @ P_hat

        V[:, unstable_mask] = (
            (bracket_plus(V_hat) * U[unstable_mask]) / (U[unstable_mask] - L[unstable_mask])
            - C[:, unstable_mask]
            - alpha * bracket_minus(V_hat)
            - pi @ P
        )

        return V, accum_sum + (
            -(bias_module.forward(V))
            + torch.sum(
                (bracket_plus(V_hat) * U[unstable_mask] * L[unstable_mask])
                / (U[unstable_mask] - L[unstable_mask]),
                dim=1,
            )
            - pi @ p
        )

    @override
    def clamp_parameters(self) -> None:
        self.pi.clamp_(min=0)
        self.alpha.clamp_(min=0, max=1)
