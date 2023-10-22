from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from preprocessing.solver_inputs import SolverInputs
from preprocessing.solver_variables import SolverVariables

from .AdversarialCheckModel import AdversarialCheckModel
from .solver_layers import SolverLayerList


class Solver(nn.Module):
    def __init__(self, inputs: SolverInputs):
        super().__init__()
        self.vars = SolverVariables(inputs)
        self.layers = SolverLayerList(self.vars)
        self.adv_check_model = AdversarialCheckModel(inputs.model, inputs.ground_truth_neuron_index)

    def reset_and_solve_for_layer(self, layer_index: int) -> None:
        self.vars.solve_for_layer(layer_index)
        self.layers = SolverLayerList(self.vars)

    def clamp_parameters(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.clamp_parameters()

    def forward(self) -> tuple[Tensor, Tensor]:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        layers = self.layers  # fmt: skip

        l = len(layers) - 1
        V: list[Tensor] = cast(list[Tensor], [None] * len(layers))
        self.V: list[Tensor] = V
        V[l] = layers[-1].forward()

        for i in range(l - 1, 0, -1):  # From l-1 to 1 (inclusive)
            V[i] = layers[i].forward(V[i + 1])

        max_objective, theta = self.compute_max_objective(V)
        return max_objective, theta

    def compute_max_objective(self, V: list[Tensor]) -> tuple[Tensor, Tensor]:
        layers, d = self.layers, self.vars.d

        l = len(layers) - 1
        theta: Tensor = layers[0].vars.C_i - V[1] @ layers[1].vars.W_i
        max_objective: Tensor = (
            (F.relu(theta) @ layers[0].vars.L_i)
            - (F.relu(-theta) @ layers[0].vars.U_i)
            + layers[-1].gamma @ d
            - torch.stack([V[i] @ layers[i].vars.b_i for i in range(1, l + 1)]).sum(dim=0)
            + torch.stack([self.layers[i].get_obj_sum() for i in range(1, l)]).sum(dim=0)
        )
        self.last_max_objective = max_objective.detach()
        return max_objective, theta

    def get_updated_bounds(self, layer_index: int) -> tuple[Tensor, Tensor]:
        """Returns `(new_lower_bounds, new_upper_bounds)` for layer `layer_index`."""
        assert self.vars.solve_coords[0][0] == layer_index

        # Clone the tensors to avoid modifying the original tensors
        new_L_i: Tensor = self.vars.L[layer_index].clone().detach()
        new_U_i: Tensor = self.vars.U[layer_index].clone().detach()

        # Iterate over the solve_coords
        for i, (_, coord) in enumerate(self.vars.solve_coords):
            # Replace bounds only if they're better than the initial bounds.
            new_L_i[coord] = torch.max(new_L_i[coord], self.last_max_objective[2 * i])
            # New upper bounds is negation of objective func.
            new_U_i[coord] = torch.min(new_U_i[coord], -self.last_max_objective[2 * i + 1])

        return new_L_i, new_U_i
