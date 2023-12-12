from typing import Tuple

import torch
from torch import Tensor, nn
from typing_extensions import override

from ...preprocessing.class_definitions import Bias
from ...preprocessing.transpose import UnaryForward
from .base_class import SolverLayer


class OutputLayer(SolverLayer):
    """The solver layer for the model's "output layer". This layer is the FIRST
    to evaluate, as the computation propagates from output-layer to
    intermediate-layers to input-layer.
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
        H: Tensor,
        d: Tensor,
    ) -> None:
        super().__init__(L, U, stably_act_mask, stably_deact_mask, unstable_mask, C)
        self.transposed_layer = transposed_layer
        self.bias_module = bias_module

        self.H: Tensor
        self.d: Tensor
        self.register_buffer("H", H)
        self.register_buffer("d", d)

    @override
    def set_C_and_reset_parameters(self, C: Tensor) -> None:
        super().set_C_and_reset_parameters(C)
        self.gamma: nn.Parameter = nn.Parameter(
            torch.rand((self.num_batches, self.H.size(0))).to(C)
        )

    def forward(self) -> Tuple[Tensor, Tensor]:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        bias_module, H, d, gamma = self.bias_module, self.H, self.d, self.gamma  # fmt: skip

        V = (-H.T @ gamma.T).T
        assert V.dim() == 2
        return V, gamma @ d - bias_module.forward(V)

    @override
    def clamp_parameters(self) -> None:
        self.gamma.clamp_(min=0)
