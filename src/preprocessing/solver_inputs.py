from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn


@dataclass
class SolverInputs:
    """Contains and validates all the raw inputs needed to start solving."""

    model: nn.Module
    ground_truth_neuron_index: int
    L_list: List[Tensor]
    U_list: List[Tensor]
    H: Tensor
    d: Tensor
    P_list: List[Tensor]
    P_hat_list: List[Tensor]
    p_list: List[Tensor]
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
        assert isinstance(self.L_list, list) and isinstance(self.L_list[0], Tensor)
        assert isinstance(self.U_list, list) and isinstance(self.U_list[0], Tensor)
        assert isinstance(self.H, Tensor)
        assert isinstance(self.d, Tensor)
        assert isinstance(self.P_list, list) and isinstance(self.P_list[0], Tensor)
        assert isinstance(self.P_hat_list, list) and isinstance(self.P_hat_list[0], Tensor)
        assert isinstance(self.p_list, list) and isinstance(self.p_list[0], Tensor)

    def _validate_tensor_dtype(self) -> None:
        EXPECT_DTYPE = torch.float
        assert self.L_list[0].dtype == EXPECT_DTYPE
        assert self.U_list[0].dtype == EXPECT_DTYPE
        assert self.H.dtype == EXPECT_DTYPE
        assert self.d.dtype == EXPECT_DTYPE
        assert self.P_list[0].dtype == EXPECT_DTYPE
        assert self.P_hat_list[0].dtype == EXPECT_DTYPE
        assert self.p_list[0].dtype == EXPECT_DTYPE

    def _validate_dimensions(self) -> None:
        for i in range(len(self.L_list)):
            assert self.L_list[i].dim() == 1
            assert self.U_list[i].dim() == 1

        assert self.H.dim() == 2
        assert self.d.dim() == 1
        assert self.H.size(0) == self.d.size(0)

        assert len(self.P_list) == len(self.P_hat_list) == len(self.p_list)
        for i in range(len(self.P_list)):
            assert self.P_list[i].dim() == self.P_hat_list[i].dim() == 2
            assert self.p_list[i].dim() == 1
            assert self.P_list[i].shape == self.P_hat_list[i].shape
            assert self.p_list[i].size(0) == self.P_list[i].size(0)

    def _validate_tensors_match_model(self) -> None:
        linear_layers = [layer for layer in self.model.children() if isinstance(layer, nn.Linear)]
        num_layers = len(linear_layers) + 1  # +1 to include input layer.
        num_neurons_per_layer: List[int] = [linear_layers[0].weight.size(1)] + [
            linear.weight.size(0) for linear in linear_layers
        ]
        assert self.ground_truth_neuron_index < num_neurons_per_layer[-1]
        assert len(self.L_list) == len(self.U_list) == num_layers == len(num_neurons_per_layer)
        for i in range(num_layers):
            assert self.L_list[i].size(0) == self.U_list[i].size(0) == num_neurons_per_layer[i]

        unstable_masks = [(self.L_list[i] < 0) & (self.U_list[i] > 0) for i in range(num_layers)]
        num_unstable_per_layer: List[int] = [int(mask.sum().item()) for mask in unstable_masks]
        num_unstable_per_intermediate_layer = num_unstable_per_layer[1:-1]
        num_intermediate_layers = num_layers - 2
        assert (
            len(self.P_list) == len(self.P_hat_list) == len(self.p_list) == num_intermediate_layers
        )
        for i in range(num_intermediate_layers):
            assert self.P_list[i].size(1) == num_unstable_per_intermediate_layer[i]

        assert self.H.size(1) == linear_layers[-1].weight.size(0)
