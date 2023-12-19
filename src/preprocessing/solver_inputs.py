import math
from typing import List, Union

import torch
from numpy import ndarray
from torch import Tensor, nn

from ..inputs.save_file_types import GurobiResults, SolverInputsSavedDict
from ..preprocessing.hwc_to_chw import (
    flattened_hwc_to_chw,
    flattened_unstable_hwc_to_chw,
)
from ..utils import load_onnx_model


class SolverInputs:
    """Contains, formats and validates all the raw inputs needed to start solving."""

    def __init__(
        self,
        model: nn.Module,
        ground_truth_neuron_index: int,
        L_list: Union[List[ndarray], List[Tensor]],
        U_list: Union[List[ndarray], List[Tensor]],
        H: Union[ndarray, Tensor],
        d: Union[ndarray, Tensor],
        P_list: Union[List[ndarray], List[Tensor]],
        P_hat_list: Union[List[ndarray], List[Tensor]],
        p_list: Union[List[ndarray], List[Tensor]],
        is_hwc: bool = True,
        skip_validation: bool = False,
    ) -> None:
        self.model: nn.Module = model
        self.ground_truth_neuron_index: int = ground_truth_neuron_index

        # Convert to tensor, float dtype, and correct dimensionality if necessary.
        self.L_list: List[Tensor] = [
            torch.atleast_1d(ensure_tensor(x).float().squeeze()) for x in L_list
        ]
        self.U_list: List[Tensor] = [
            torch.atleast_1d(ensure_tensor(x).float().squeeze()) for x in U_list
        ]
        self.H: Tensor = torch.atleast_2d(ensure_tensor(H).float().squeeze())
        self.d: Tensor = torch.atleast_1d(ensure_tensor(d).float().squeeze())
        self.P_list: List[Tensor] = [
            torch.atleast_2d(ensure_tensor(x).float().squeeze()) for x in P_list
        ]
        self.P_hat_list: List[Tensor] = [
            torch.atleast_2d(ensure_tensor(x).float().squeeze()) for x in P_hat_list
        ]
        self.p_list: List[Tensor] = [
            torch.atleast_1d(ensure_tensor(x).float().squeeze()) for x in p_list
        ]

        if is_hwc:
            self._convert_hwc_to_chw()
        if skip_validation:
            return
        self._validate_types()
        self._validate_tensor_dtype()
        self._validate_dimensions()

    def save_all_except_model(self, save_file_path: str) -> None:
        """Saves all the inputs except the model.

        Args:
            save_file_path (str): Path to save the inputs to.
        """
        saved_dict: SolverInputsSavedDict = {
            "L_list": self.L_list,
            "U_list": self.U_list,
            "H": self.H,
            "d": self.d,
            "P_list": self.P_list,
            "P_hat_list": self.P_hat_list,
            "p_list": self.p_list,
            "ground_truth_neuron_index": self.ground_truth_neuron_index,
            "is_hwc": False,
        }
        torch.save(saved_dict, save_file_path)

    @staticmethod
    def load(onnx_model_path: str, other_inputs_path: str) -> "SolverInputs":
        """Load ONNX model and the other inputs (saved in `SolverInputsSavedDict` format)
        from their save files.

        Args:
            onnx_model_path (str): Path to ONNX model save file.
            other_inputs_path (str): Path to the other inputs (saved in \
                `SolverInputsSavedDict` format).
        """
        model: nn.Module = load_onnx_model(onnx_model_path)
        loaded: SolverInputsSavedDict = torch.load(other_inputs_path)
        return SolverInputs(model=model, **loaded)

    def convert_gurobi_hwc_to_chw(
        self,
        gurobi_results: GurobiResults,
        hwc_L_list: List[Tensor],
        hwc_U_list: List[Tensor],
    ) -> GurobiResults:
        """Converts Gurobi-computed HWC-formatted unstable-only bounds to
        CHW-format.

        HWC: Height-Width-Channel
        CHW: Channel-Height-Width

        Args:
            gurobi_results (GurobiResults): Gurobi-computed HWC-formatted unstable-only bounds.
            hwc_L_list (List[Tensor]): Original full (ie. all-neurons) lower-bounds in HWC-format.
            hwc_U_list (List[Tensor]): Original full (ie. all-neurons) upper-bounds in HWC-format.

        Returns:
            GurobiResults: Gurobi's results but in CHW-format.
        """
        hwc_unstable_masks = [(L < 0) & (U > 0) for L, U in zip(hwc_L_list, hwc_U_list)]

        gurobi_L_list = list(gurobi_results["L_list_unstable_only"])
        gurobi_U_list = list(gurobi_results["U_list_unstable_only"])

        first_layer = next(self.model.children())

        if isinstance(first_layer, nn.Conv2d):
            num_neurons = self.L_list[0].size(0)
            num_channels = first_layer.in_channels

            # Assume that `height == width` for the CNN input.
            H_W = int(math.sqrt(num_neurons / num_channels))
            hwc_shape = (H_W, H_W, num_channels)

            gurobi_L_list[0] = flattened_hwc_to_chw(gurobi_L_list[0], hwc_shape)
            gurobi_U_list[0] = flattened_hwc_to_chw(gurobi_U_list[0], hwc_shape)

        i = 1
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                i += 1
                continue
            if not isinstance(layer, nn.Conv2d):
                continue

            num_neurons = self.L_list[i].size(0)
            num_channels = layer.out_channels

            # Assumes that `height == width` for all CNN inputs.
            H_W = int(math.sqrt(num_neurons / num_channels))
            hwc_shape = (H_W, H_W, num_channels)

            gurobi_L_list[i] = flattened_unstable_hwc_to_chw(
                gurobi_L_list[i],
                hwc_unstable_masks[i],
                hwc_shape,
            )

            gurobi_U_list[i] = flattened_unstable_hwc_to_chw(
                gurobi_U_list[i],
                hwc_unstable_masks[i],
                hwc_shape,
            )
            i += 1

        return {
            "L_list_unstable_only": gurobi_L_list,
            "U_list_unstable_only": gurobi_U_list,
            "compute_time": gurobi_results["compute_time"],
        }

    def _convert_hwc_to_chw(self) -> None:
        """Converts the tensor inputs from Height-Width-Channel (HWC) format to
        the Channel-Height-Width (CHW) format that Pytorch expects.

        Warning: Assumes that `height == width` for all CNN inputs.
        """
        first_layer = next(self.model.children())

        if isinstance(first_layer, nn.Conv2d):
            num_neurons = self.L_list[0].size(0)
            num_channels = first_layer.in_channels

            # Assume that `height == width` for the CNN input.
            H_W = int(math.sqrt(num_neurons / num_channels))
            hwc_shape = (H_W, H_W, num_channels)

            self.L_list[0] = flattened_hwc_to_chw(self.L_list[0], hwc_shape)
            self.U_list[0] = flattened_hwc_to_chw(self.U_list[0], hwc_shape)

        i = 1
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                i += 1
                continue
            if not isinstance(layer, nn.Conv2d):
                continue

            num_neurons = self.L_list[i].size(0)
            num_channels = layer.out_channels

            # Assumes that `height == width` for all CNN inputs.
            H_W = int(math.sqrt(num_neurons / num_channels))
            hwc_shape = (H_W, H_W, num_channels)

            unstable_mask = (self.L_list[i] < 0) & (self.U_list[i] > 0)
            self.L_list[i] = flattened_hwc_to_chw(self.L_list[i], hwc_shape)
            self.U_list[i] = flattened_hwc_to_chw(self.U_list[i], hwc_shape)

            is_last_layer = i >= len(self.P_list) + 1
            if is_last_layer:
                continue

            self.P_list[i - 1] = flattened_unstable_hwc_to_chw(
                self.P_list[i - 1],
                unstable_mask,
                hwc_shape,
                mask_dim=1,
            )
            self.P_hat_list[i - 1] = flattened_unstable_hwc_to_chw(
                self.P_hat_list[i - 1],
                unstable_mask,
                hwc_shape,
                mask_dim=1,
            )
            i += 1
            continue

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


def ensure_tensor(array_or_tensor: Union[ndarray, Tensor]) -> Tensor:
    """Converts `array_or_tensor` to a Pytorch Tensor if necessary."""
    return (
        torch.from_numpy(array_or_tensor)
        if isinstance(array_or_tensor, ndarray)
        else array_or_tensor
    )
