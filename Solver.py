from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from inputs_dataclasses import SolverInputs
from SolverLayerList import SolverLayerList


class Solver(nn.Module):
    def __init__(self, inputs: SolverInputs):
        super().__init__()
        self.d = inputs.d
        self.layers = SolverLayerList(inputs)

    def set_target(self, i: int, j: int, is_min) -> None:
        for layer in self.layers:
            layer.clear_C()
        self.layers[i].set_C(j, is_min)

    def clamp_parameters(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.clamp_parameters()

    def forward(self) -> Tensor:
        layers = self.layers

        l = layers.num_layers
        V: list[Tensor] = cast(list[Tensor], [None] * len(layers))
        self.V: list[Tensor] = V
        V[l] = layers[-1].forward()

        for i in range(l - 1, 0, -1):  # From l-1 to 1 (inclusive)
            V[i] = layers[i].forward(V[i + 1])

        loss = -self.compute_max_objective(V)
        return loss.sum()

    def compute_max_objective(self, V: list[Tensor]) -> Tensor:
        layers, d = self.layers, self.d

        l = layers.num_layers
        relu_tensor: Tensor = layers[0].C_i.T - V[1].T @ layers[1].inputs.W_i
        max_objective: Tensor = (
            (F.relu(relu_tensor) @ layers[0].inputs.L_i)
            - (F.relu(-relu_tensor) @ layers[0].inputs.U_i)
            + layers[-1].gamma.T @ d
            - torch.stack([V[i].T @ layers[i].inputs.b_i for i in range(1, l + 1)]).sum(dim=0)
            + torch.stack([self.layers[i].get_obj_sum() for i in range(1, l)]).sum(dim=0)
        )
        return max_objective
