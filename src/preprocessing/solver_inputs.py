from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn


@dataclass
class SolverInputs:
    """Contains and validates all the raw inputs needed to start solving."""

    model: nn.Module
    ground_truth_neuron_index: int
    L: List[Tensor]
    U: List[Tensor]
    H: Tensor
    d: Tensor
    P: List[Tensor]
    P_hat: List[Tensor]
    p: List[Tensor]
    skip_validation: bool = False

    def __post_init__(self) -> None:
        if self.skip_validation:
            return
        self._validate_types()
        self._validate_tensor_dtype()
        self._validate_dimensions()
        self._validate_tensors_match_model()

    def _validate_types(self) -> None:
        assert isinstance(self.model, nn.Module)
        assert isinstance(self.ground_truth_neuron_index, int)
        assert isinstance(self.L, list) and isinstance(self.L[0], Tensor)
        assert isinstance(self.U, list) and isinstance(self.U[0], Tensor)
        assert isinstance(self.H, Tensor)
        assert isinstance(self.d, Tensor)
        assert isinstance(self.P, list) and isinstance(self.P[0], Tensor)
        assert isinstance(self.P_hat, list) and isinstance(self.P_hat[0], Tensor)
        assert isinstance(self.p, list) and isinstance(self.p[0], Tensor)

    def _validate_tensor_dtype(self) -> None:
        EXPECT_DTYPE = torch.float
        assert self.L[0].dtype == EXPECT_DTYPE
        assert self.U[0].dtype == EXPECT_DTYPE
        assert self.H.dtype == EXPECT_DTYPE
        assert self.d.dtype == EXPECT_DTYPE
        assert self.P[0].dtype == EXPECT_DTYPE
        assert self.P_hat[0].dtype == EXPECT_DTYPE
        assert self.p[0].dtype == EXPECT_DTYPE

    def _validate_dimensions(self) -> None:
        for i in range(len(self.L)):
            assert self.L[i].dim() == 1
            assert self.U[i].dim() == 1

        assert self.H.dim() == 2
        assert self.d.dim() == 1
        assert self.H.size(0) == self.d.size(0)

        assert len(self.P) == len(self.P_hat) == len(self.p)
        for i in range(len(self.P)):
            assert self.P[i].dim() == self.P_hat[i].dim() == 2
            assert self.p[i].dim() == 1
            assert self.P[i].shape == self.P_hat[i].shape
            assert self.p[i].size(0) == self.P[i].size(0)

    def _validate_tensors_match_model(self) -> None:
        linear_layers = [layer for layer in self.model.children() if isinstance(layer, nn.Linear)]
        num_layers = len(linear_layers) + 1  # +1 to include input layer.
        num_neurons_per_layer: List[int] = [linear_layers[0].weight.size(1)] + [
            linear.weight.size(0) for linear in linear_layers
        ]
        assert self.ground_truth_neuron_index < num_neurons_per_layer[-1]
        assert len(self.L) == len(self.U) == num_layers == len(num_neurons_per_layer)
        for i in range(num_layers):
            assert self.L[i].size(0) == self.U[i].size(0) == num_neurons_per_layer[i]

        unstable_masks = [(self.L[i] < 0) & (self.U[i] > 0) for i in range(num_layers)]
        num_unstable_per_layer: List[int] = [int(mask.sum().item()) for mask in unstable_masks]
        num_unstable_per_intermediate_layer = num_unstable_per_layer[1:-1]
        num_intermediate_layers = num_layers - 2
        assert len(self.P) == len(self.P_hat) == len(self.p) == num_intermediate_layers
        for i in range(num_intermediate_layers):
            assert self.P[i].size(1) == num_unstable_per_intermediate_layer[i]

        assert self.H.size(1) == linear_layers[-1].weight.size(0)
