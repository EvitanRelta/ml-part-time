from typing import Tuple

import torch
from torch import Tensor, nn
from typing_extensions import override


class Output_SL(nn.Module):
    """The solver layer for the model's "output layer". This layer is the FIRST
    to evaluate, as the computation propagates from output-layer to
    intermediate-layers to input-layer.
    """

    @override
    def __init__(
        self,
        num_batches: int,
        H: Tensor,
        d: Tensor,
    ) -> None:
        super().__init__()
        self.num_batches = num_batches

        self.H: Tensor
        self.d: Tensor
        self.register_buffer("H", H)
        self.register_buffer("d", d)

    def reset_parameters(self, num_batches: int) -> None:
        self.num_batches = num_batches
        self.gamma = nn.Parameter(torch.rand((self.num_batches, self.H.size(0))).to(self.H))

    def forward(self) -> Tuple[Tensor, Tensor, Tensor]:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        H, d, gamma = self.H, self.d, self.gamma

        V = (-H.T @ gamma.T).T
        assert V.dim() == 2
        return V, torch.zeros((self.num_batches,)), gamma @ d

    def clamp_parameters(self) -> None:
        self.gamma.clamp_(min=0)
