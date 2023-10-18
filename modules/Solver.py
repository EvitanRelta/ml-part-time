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

    def forward(self) -> Tensor:
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        layers = self.layers  # fmt: skip

        l = len(layers) - 1
        V: list[Tensor] = cast(list[Tensor], [None] * len(layers))
        self.V: list[Tensor] = V
        V[l] = layers[-1].forward()

        for i in range(l - 1, 0, -1):  # From l-1 to 1 (inclusive)
            V[i] = layers[i].forward(V[i + 1])

        loss = -self.compute_max_objective(V)
        return loss.sum()

    def compute_max_objective(self, V: list[Tensor]) -> Tensor:
        layers, d = self.layers, self.vars.d

        l = len(layers) - 1
        relu_tensor: Tensor = layers[0].vars.C_i - V[1] @ layers[1].vars.W_i
        max_objective: Tensor = (
            (F.relu(relu_tensor) @ layers[0].vars.L_i)
            - (F.relu(-relu_tensor) @ layers[0].vars.U_i)
            + layers[-1].gamma @ d
            - torch.stack([V[i] @ layers[i].vars.b_i for i in range(1, l + 1)]).sum(dim=0)
            + torch.stack([self.layers[i].get_obj_sum() for i in range(1, l)]).sum(dim=0)
        )
        return max_objective
