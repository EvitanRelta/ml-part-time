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
        # fmt: off
        error_msg = "Expected tensor `{var}` to be of dtype=" + str(EXPECT_DTYPE) +", but got `{list[0].dtype}`."
        list_error_msg = "Expected all tensors in `{var}` to be of dtype=" + str(EXPECT_DTYPE) +", but got `{tensor.dtype}`."
        assert self.L_list[0].dtype == EXPECT_DTYPE, list_error_msg.format(var='L_list', list=self.L_list)
        assert self.U_list[0].dtype == EXPECT_DTYPE, list_error_msg.format(var='U_list', list=self.U_list)
        assert self.H.dtype == EXPECT_DTYPE, error_msg.format(var='H', tensor=self.H)
        assert self.d.dtype == EXPECT_DTYPE, error_msg.format(var='d', tensor=self.d)
        assert self.P_list[0].dtype == EXPECT_DTYPE, list_error_msg.format(var='P_list', list=self.P_list)
        assert self.P_hat_list[0].dtype == EXPECT_DTYPE, list_error_msg.format(var='P_hat_list', list=self.P_hat_list)
        assert self.p_list[0].dtype == EXPECT_DTYPE, list_error_msg.format(var='p_list', list=self.p_list)
        # fmt: on

    def _validate_dimensions(self) -> None:
        # fmt: off
        error_msg = "Expected tensor `{var}` to be {expected_dim}D, but got {dim}D."
        for i in range(len(self.L_list)):
            assert self.L_list[i].dim() == 1, error_msg.format(var=f'L_list[{i}]', expected_dim=1, dim=self.L_list[i].dim())
            assert self.U_list[i].dim() == 1, error_msg.format(var=f'U_list[{i}]', expected_dim=1, dim=self.U_list[i].dim())

        assert self.H.dim() == 2, error_msg.format(var="H", expected_dim=2, dim=self.H.dim())
        assert self.d.dim() == 1, error_msg.format(var="d", expected_dim=1, dim=self.d.dim())
        assert self.H.size(0) == self.d.size(0), f"Expected len(H) == len(d), but got {self.H.size(0)} == {self.d.size(0)}."

        assert len(self.P_list) == len(self.P_hat_list) == len(self.p_list), f"Expected len(P_list) == len(P_hat_list) == len(p_list), but got {len(self.P_list)} == {len(self.P_hat_list)} == {len(self.p_list)}."

        for i in range(len(self.P_list)):
            assert self.P_list[i].dim() == 2, error_msg.format(var=f'P_list[{i}]', expected_dim=2, dim=self.P_list[i].dim())
            assert self.P_hat_list[i].dim() == 2, error_msg.format(var=f'P_hat_list[{i}]', expected_dim=2, dim=self.P_hat_list[i].dim())
            assert self.p_list[i].dim() == 1, error_msg.format(var=f'p_list[{i}]', expected_dim=1, dim=self.p_list[i].dim())
            assert self.P_list[i].shape == self.P_hat_list[i].shape, f"Expected `P_list[{i}]` and `P_hat_list[{i}]` to be of same shape, but got {tuple(self.P_list[i].shape)} and {tuple(self.P_hat_list[i].shape)} respectively."
            assert self.p_list[i].size(0) == self.P_list[i].size(0), f"Expected len(p_list[{i}]) == len(P_list[{i}]), but got {self.p_list[i].size(0)} == {self.P_list[i].size(0)}."
        # fmt: on

    def _validate_tensors_match_model(self) -> None:
        linear_layers = [layer for layer in self.model.children() if isinstance(layer, nn.Linear)]
        num_layers = len(linear_layers) + 1  # +1 to include input layer.
        num_neurons_per_layer: List[int] = [linear_layers[0].weight.size(1)] + [
            linear.weight.size(0) for linear in linear_layers
        ]
        # fmt: off
        assert num_layers == len(num_neurons_per_layer), "This shouldn't happen."
        assert 0 <= self.ground_truth_neuron_index < num_neurons_per_layer[-1], f"Expected 0 <= ground_truth_neuron_index < {num_neurons_per_layer[-1]}, but got {self.ground_truth_neuron_index} ({num_neurons_per_layer[-1]} is the num of neurons in the output layer)."
        assert len(self.L_list) == len(self.U_list) == num_layers, f"Expected len(L_list) == len(U_list) == num of linear layers in `model` + 1, but got {len(self.L_list)} == {len(self.U_list)} == {num_layers}."
        # fmt: on
        for i in range(num_layers):
            assert self.L_list[i].size(0) == self.U_list[i].size(0) == num_neurons_per_layer[i]

        unstable_masks = [(self.L_list[i] < 0) & (self.U_list[i] > 0) for i in range(num_layers)]
        num_unstable_per_layer: List[int] = [int(mask.sum().item()) for mask in unstable_masks]
        num_unstable_per_intermediate_layer = num_unstable_per_layer[1:-1]
        num_intermediate_layers = num_layers - 2

        # fmt: off
        ith = lambda i: "1st" if i == 1 \
            else "2nd" if i == 2 \
            else "3rd" if i == 3 \
            else f"{i}th"
        assert len(self.P_list) == len(self.P_hat_list) == len(self.p_list) == num_intermediate_layers, f"Expected len(P_list) == len(P_hat_list) == len(p_list) == num of intermediate layers, but got {len(self.P_list)} == {len(self.P_hat_list)} == {len(self.p_list)}."
        for i in range(num_intermediate_layers):
            assert self.P_list[i].size(1) == num_unstable_per_intermediate_layer[i], f"Expected P_list[{i}].size(1) == {num_unstable_per_intermediate_layer[i]}, but got {self.P_list[i].size(1)} ({num_unstable_per_intermediate_layer[i]} is the num of unstable neurons in the {ith(i)} intermediate layer)."
        # fmt: on

        assert self.H.size(1) == linear_layers[-1].weight.size(0)
