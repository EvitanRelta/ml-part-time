import torch
from torch import Tensor, nn
from typing_extensions import override

from ...preprocessing.solver_variables import OutputLayerVariables
from .base_class import SolverLayer


class SolverOutput(SolverLayer):
    @override
    def __init__(self, vars: OutputLayerVariables) -> None:
        super().__init__(vars)
        self.vars: OutputLayerVariables
        self.gamma: nn.Parameter = nn.Parameter(
            torch.rand((self.vars.num_batches, self.vars.H.size(0)))
        )

    @override
    def forward(self, V_next: Tensor = torch.empty(0)) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        H, gamma = self.vars.H, self.gamma  # fmt: skip

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
