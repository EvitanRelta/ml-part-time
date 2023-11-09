from typing import Tuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from ...preprocessing.transpose import UnaryForward
from .base_class import SolverLayer


class SolverOutput(SolverLayer):
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
        H: Tensor,
        d: Tensor,
    ) -> None:
        super().__init__(L, U, stably_act_mask, stably_deact_mask, unstable_mask, C)
        self.transposed_layer = transposed_layer

        self.b: Tensor
        self.H: Tensor
        self.d: Tensor
        self.register_buffer("b", b)
        self.register_buffer("H", H)
        self.register_buffer("d", d)

    @override
    def set_C_and_reset(self, C: Tensor) -> None:
        super().set_C_and_reset(C)
        self.gamma: nn.Parameter = nn.Parameter(torch.rand((self.num_batches, self.H.size(0))))

    def forward(self) -> Tuple[Tensor, Tensor]:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        H, d, gamma = self.H, self.d, self.gamma  # fmt: skip

        V = (-H.T @ gamma.T).T
        assert V.dim() == 2
        return V, gamma @ d - V @ self.b

    @override
    def clamp_parameters(self) -> None:
        self.gamma.clamp_(min=0)
