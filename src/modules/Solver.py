from typing import List, Tuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..preprocessing.solver_inputs import SolverInputs
from .AdversarialCheckModel import AdversarialCheckModel
from .SolverLayerList import SolverLayerList


class Solver(nn.Module):
    def __init__(self, inputs: SolverInputs):
        super().__init__()
        self.layers = SolverLayerList(inputs)
        self.adv_check_model = AdversarialCheckModel(inputs.model, inputs.ground_truth_neuron_index)

    def reset_and_solve_for_layer(self, layer_index: int) -> None:
        self.layers.solve_for_layer(layer_index)

    def clamp_parameters(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.clamp_parameters()

    def forward(self) -> Tuple[Tensor, Tensor]:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        layers = self.layers  # fmt: skip

        l = len(layers) - 1
        V_list: List[Tensor] = cast(List[Tensor], [None] * len(layers))
        self.V_list: List[Tensor] = V_list
        V_list[l] = layers[-1].forward()

        for i in range(l - 1, 0, -1):  # From l-1 to 1 (inclusive)
            V_list[i] = layers[i].forward(V_list[i + 1])

        max_objective, theta = self.compute_max_objective(V_list)
        return max_objective, theta

    def compute_max_objective(self, V_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
        layers, d = self.layers, self.layers.d

        l = len(layers) - 1
        theta: Tensor = layers[0].C - layers[1].transposed_layer.forward(V_list[1])
        max_objective: Tensor = (
            (F.relu(theta) @ layers[0].L)
            - (F.relu(-theta) @ layers[0].U)
            + layers[-1].gamma @ d
            - torch.stack([V_list[i] @ layers[i].b for i in range(1, l + 1)]).sum(dim=0)
            + torch.stack([self.layers[i].get_obj_sum() for i in range(1, l)]).sum(dim=0)
        )
        self.last_max_objective = max_objective.detach()
        return max_objective, theta.detach()

    def get_updated_bounds(self, layer_index: int) -> Tuple[Tensor, Tensor]:
        """Returns `(new_lower_bounds, new_upper_bounds)` for layer `layer_index`."""
        assert self.layers.solve_coords[0][0] == layer_index

        # Clone the tensors to avoid modifying the original tensors
        new_L: Tensor = self.layers.L_list[layer_index].clone().detach()
        new_U: Tensor = self.layers.U_list[layer_index].clone().detach()

        # Iterate over the solve_coords
        for i, (_, coord) in enumerate(self.layers.solve_coords):
            # Replace bounds only if they're better than the initial bounds.
            new_L[coord] = torch.max(new_L[coord], self.last_max_objective[2 * i])
            # New upper bounds is negation of objective func.
            new_U[coord] = torch.min(new_U[coord], -self.last_max_objective[2 * i + 1])

        return new_L, new_U
