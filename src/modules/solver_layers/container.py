from typing import List, Tuple

import torch
from torch import Tensor, nn

from ...preprocessing import preprocessing_utils
from ...preprocessing.build import build_solver_graph_module
from ...preprocessing.solver_inputs import SolverInputs
from .base_class import Solvable_SL
from .output_layer import Output_SL


class SolverLayerContainer(nn.Module):
    """Module containing all the solver layers."""

    def __init__(self, inputs: SolverInputs) -> None:
        super().__init__()
        self.graph_module = build_solver_graph_module(inputs)

    def solve_for_layer(self, layer_index: int) -> None:
        C_list, self.solve_coords = preprocessing_utils.get_C_for_layer(
            layer_index, self.unstable_masks
        )
        for solvable_layer, C in zip(self.solvable_layers, C_list):
            solvable_layer.set_C_and_reset_parameters(C)

        self.output_layer.reset_parameters(num_batches=C_list[0].size(0))

    def forward(self) -> Tuple[Tensor, Tensor]:
        return self.graph_module.forward()

    def clamp_parameters(self):
        with torch.no_grad():
            for layer in self.solvable_layers:
                layer.clamp_parameters()

            self.output_layer.clamp_parameters()

    @property
    def solvable_layers(self) -> List[Solvable_SL]:
        if not hasattr(self, "_solvable_layers"):
            self._solvable_layers: List[Solvable_SL] = [
                x for x in self.graph_module.children() if isinstance(x, Solvable_SL)
            ]
            self._solvable_layers.reverse()
        return self._solvable_layers

    @property
    def output_layer(self) -> Output_SL:
        output_layer = self.graph_module.get_submodule("output_layer")
        assert isinstance(output_layer, Output_SL)
        return output_layer

    @property
    def L_list(self) -> List[Tensor]:
        return [x.L for x in self.solvable_layers]

    @property
    def U_list(self) -> List[Tensor]:
        return [x.U for x in self.solvable_layers]

    @property
    def H(self) -> Tensor:
        return self.output_layer.H

    @property
    def stably_act_masks(self) -> List[Tensor]:
        return [x.stably_act_mask for x in self.solvable_layers]

    @property
    def stably_deact_masks(self) -> List[Tensor]:
        return [x.stably_deact_mask for x in self.solvable_layers]

    @property
    def unstable_masks(self) -> List[Tensor]:
        return [x.unstable_mask for x in self.solvable_layers]

    @property
    def C_list(self) -> List[Tensor]:
        return [x.C for x in self.solvable_layers]
