from typing import Iterator, List, Literal, overload

from torch import nn

from ..preprocessing.solver_variables import SolverVariables
from .solver_layers.base_class import SolverLayer
from .solver_layers.SolverInput import SolverInput
from .solver_layers.SolverIntermediate import SolverIntermediate
from .solver_layers.SolverOutput import SolverOutput


class SolverLayerList(nn.ModuleList):
    """Wrapper around `ModuleList` to contain `SolverLayer` modules."""

    def __init__(self, vars: SolverVariables):
        layers: List[SolverLayer] = [SolverInput(vars.layer_vars[0])]
        for i in range(1, len(vars.layer_vars) - 1):
            layers.append(SolverIntermediate(vars.layer_vars[i]))
        layers.append(SolverOutput(vars.layer_vars[-1]))
        super().__init__(layers)

    def __iter__(self) -> Iterator["SolverLayer"]:
        return super().__iter__()  # type: ignore

    # fmt: off
    @overload
    def __getitem__(self, i: Literal[0]) -> "SolverInput": ...
    @overload
    def __getitem__(self, i: Literal[-1]) -> "SolverOutput": ...
    @overload
    def __getitem__(self, i: int) -> "SolverIntermediate": ...
    # fmt: on
    def __getitem__(self, i: int) -> "SolverLayer":
        return super().__getitem__(i)  # type: ignore
