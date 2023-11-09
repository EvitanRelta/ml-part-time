from torch import Tensor
from typing_extensions import override

from .base_class import LayerVariables, SolverLayer


class InputLayerVariables(LayerVariables):
    ...


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
