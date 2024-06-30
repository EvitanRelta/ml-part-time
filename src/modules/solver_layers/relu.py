from typing import Tuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from ..solver_utils import bracket_minus, bracket_plus
from .base_class import Solvable_SL


class ReLU_SL(Solvable_SL):
    """The solver layer for the model's "intermediate layers". These layers are
    evaluated after the solver-output-layer, as the computation propagates
    from output-layer to intermediate-layers to input-layer.
    """

    @override
    def __init__(
        self,
        L: Tensor,
        U: Tensor,
        C: Tensor,
        P: Tensor,
        P_hat: Tensor,
        p: Tensor,
    ) -> None:
        super().__init__(L, U, C)

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

    def forward(
        self,
        *args: Tuple[Tensor, Tensor, Tensor],
        set_zero_accum: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        V_list, V_W_list, accum_sum_list = zip(*args)
        V_W_next = torch.stack(V_W_list).sum(dim=0)

        if set_zero_accum:
            assert len(accum_sum_list) == 1
            accum_sum = torch.zeros_like(accum_sum_list[0])
        else:
            accum_sum = torch.stack(accum_sum_list).sum(dim=0)

        # Assign to local variables, so that they can be used w/o `self.` prefix.
        num_batches, layer_shape, num_unstable, P, P_hat, p, C, stably_act_mask, stably_deact_mask, unstable_mask, pi, alpha, U, L = self.num_batches, self.layer_shape, self.num_unstable, self.P, self.P_hat, self.p, self.C, self.stably_act_mask, self.stably_deact_mask, self.unstable_mask, self.pi, self.alpha, self.U, self.L  # fmt: skip
        device = V_W_next.device

        V: Tensor = torch.zeros((num_batches, *layer_shape)).to(device)

        # Stably activated.
        stably_activated_V: Tensor = V_W_next - C
        V[:, stably_act_mask] = stably_activated_V[:, stably_act_mask]

        # Stably deactivated.
        V[:, stably_deact_mask] = -C[:, stably_deact_mask]

        # Unstable.
        if num_unstable == 0:
            # `V_W` is undefined for ReLU layers, and won't be used by next layer.
            # Thus return a placeholder zero-tensor as `V_W`.
            return V, torch.zeros(0), accum_sum - pi @ p

        V_hat = V_W_next[:, unstable_mask] - pi @ P_hat

        V[:, unstable_mask] = (
            (bracket_plus(V_hat) * U[unstable_mask]) / (U[unstable_mask] - L[unstable_mask])
            - C[:, unstable_mask]
            - alpha * bracket_minus(V_hat)
            - pi @ P
        )

        return (
            V,
            V,
            accum_sum
            + torch.sum(
                (bracket_plus(V_hat) * U[unstable_mask] * L[unstable_mask])
                / (U[unstable_mask] - L[unstable_mask]),
                dim=1,
            )
            - pi @ p,
        )

    @override
    def clamp_parameters(self) -> None:
        self.pi.clamp_(min=0)
        self.alpha.clamp_(min=0, max=1)
