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
    ) -> None:
        super().__init__(L, U, stably_act_mask, stably_deact_mask, unstable_mask, C)
        self.transposed_layer = transposed_layer

        self.b: Tensor
        self.H: Tensor
        self.register_buffer("b", b)
        self.register_buffer("H", H)

    @override
    def set_C_and_reset(self, C: Tensor) -> None:
        super().set_C_and_reset(C)
        self.gamma: nn.Parameter = nn.Parameter(torch.rand((self.num_batches, self.H.size(0))))

    @override
    def forward(self, V_next: Tensor = torch.empty(0)) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        H, gamma = self.H, self.gamma  # fmt: skip

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
