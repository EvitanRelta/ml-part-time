from abc import ABC, abstractmethod

from torch import Tensor, nn

from ...preprocessing.solver_variables import LayerVariables


class SolverLayer(ABC, nn.Module):
    """Abstract base class for all solver layers."""

    def __init__(self, vars: LayerVariables) -> None:
        super().__init__()
        self.vars = vars

    # fmt: off
    @abstractmethod
    def forward(self, V_next: Tensor) -> Tensor: ...

    @abstractmethod
    def clamp_parameters(self) -> None: ...

    @abstractmethod
    def get_obj_sum(self) -> Tensor: ...
    # fmt: on
