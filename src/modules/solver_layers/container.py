from typing import Iterator, List, Literal, Tuple, overload

import torch
from torch import Tensor, nn

from ...preprocessing import preprocessing_utils
from ...preprocessing.build import build_solver_graph_module
from ...preprocessing.solver_inputs import SolverInputs
from .base_class import SolverLayer
from .input_layer import InputLayer
from .intermediate_layer import IntermediateLayer
from .output_layer import OutputLayer


class SolverLayerContainer(nn.Module):
    """Module containing all the solver layers."""

    def __init__(self, inputs: SolverInputs) -> None:
        super().__init__()
        self.graph_module = build_solver_graph_module(inputs)

    def solve_for_layer(self, layer_index: int) -> None:
        C_list, self.solve_coords = preprocessing_utils.get_C_for_layer(
            layer_index, self.unstable_masks
        )
        for i in range(len(self)):
            self[i].set_C_and_reset_parameters(C_list[i])

    def forward(self) -> Tuple[Tensor, Tensor]:
        return self.graph_module.forward()

    def clamp_parameters(self):
        with torch.no_grad():
            for layer in self:
                layer.clamp_parameters()

    @property
    def _solver_layers(self) -> List[SolverLayer]:
        if not hasattr(self, "__solver_layers"):
            self.__solver_layers: List[SolverLayer] = [
                x for x in self.graph_module.children() if isinstance(x, SolverLayer)
            ]
            self.__solver_layers.reverse()
        return self.__solver_layers

    def __len__(self) -> int:
        return len(self._solver_layers)

    def __iter__(self) -> Iterator[SolverLayer]:
        return iter(self._solver_layers)

    # fmt: off
    @overload
    def __getitem__(self, i: Literal[0]) -> InputLayer: ...
    @overload
    def __getitem__(self, i: Literal[-1]) -> OutputLayer: ...
    @overload
    def __getitem__(self, i: int) -> IntermediateLayer: ...
    # fmt: on
    def __getitem__(self, i: int) -> SolverLayer:
        return self._solver_layers[i]

    @property
    def L_list(self) -> List[Tensor]:
        return [x.L for x in self]

    @property
    def U_list(self) -> List[Tensor]:
        return [x.U for x in self]

    @property
    def H(self) -> Tensor:
        return self[-1].H

    @property
    def stably_act_masks(self) -> List[Tensor]:
        return [x.stably_act_mask for x in self]

    @property
    def stably_deact_masks(self) -> List[Tensor]:
        return [x.stably_deact_mask for x in self]

    @property
    def unstable_masks(self) -> List[Tensor]:
        return [x.unstable_mask for x in self]

    @property
    def C_list(self) -> List[Tensor]:
        return [x.C for x in self]
