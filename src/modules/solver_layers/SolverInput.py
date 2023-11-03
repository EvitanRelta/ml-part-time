from torch import Tensor
from typing_extensions import override

from ...preprocessing.solver_variables import InputLayerVariables
from .base_class import SolverLayer


class SolverInput(SolverLayer):
    @override
    def __init__(self, vars: InputLayerVariables) -> None:
        super().__init__(vars)
        self.vars: InputLayerVariables

    # fmt: off
    @override
    def forward(self, V_next: Tensor) -> Tensor: raise NotImplementedError()

    @override
    def clamp_parameters(self) -> None: ...

    @override
    def get_obj_sum(self) -> Tensor: raise NotImplementedError()
    # fmt: on
