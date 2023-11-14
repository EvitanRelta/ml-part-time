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
        error_msg = "Expected tensor `{var}` to be of dtype=" + str(EXPECT_DTYPE) +", but got `{tensor.dtype}`."
        list_error_msg = "Expected all tensors in `{var}` to be of dtype=" + str(EXPECT_DTYPE) +", but got `{list[0].dtype}`."
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
