import functools
from typing import List, Sequence, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor, fx, nn

from ..inputs.save_file_types import GurobiResults, SolverInputsSavedDict
from ..preprocessing.graph_module_wrapper import GraphModuleWrapper
from ..preprocessing.hwc_to_chw import (
    flattened_hwc_to_chw,
    flattened_unstable_hwc_to_chw,
)
from ..preprocessing.preprocessing_utils import (
    remove_onnx_norm_layers,
    replace_reshape_with_flatten,
)
from ..utils import load_onnx_model


class SolverInputs:
    """Contains, formats and validates all the raw inputs needed to start solving."""

    def __init__(
        self,
        model: fx.GraphModule,
        input_shape: Tuple[int, ...],
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
        self.model = model
        replace_reshape_with_flatten(self.model)
        remove_onnx_norm_layers(self.model)
        self.input_shape: Tuple[int, ...] = input_shape
        self.graph_wrapper = GraphModuleWrapper(self.model, self.input_shape)
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
        self._mask_p_for_unstable_only()

        if not skip_validation:
            self._validate_tensors_match_model()
            self._validate_types()
            self._validate_tensor_dtype()
            self._validate_dimensions()

        self._unflatten_bounds(is_hwc)

        if is_hwc:
            self._convert_hwc_to_chw()

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
        model, input_shape = load_onnx_model(onnx_model_path, return_input_shape=True)
        loaded: SolverInputsSavedDict = torch.load(other_inputs_path)
        return SolverInputs(model, input_shape, **loaded)

    def convert_gurobi_hwc_to_chw(self, gurobi_results: GurobiResults) -> GurobiResults:
        """Converts Gurobi-computed HWC-formatted unstable-only bounds to CHW-format.

        HWC: Height-Width-Channel
        CHW: Channel-Height-Width

        Args:
            gurobi_results (GurobiResults): Gurobi-computed HWC-formatted unstable-only bounds.

        Returns:
            GurobiResults: Gurobi's results but in CHW-format.
        """
        hwc_unstable_masks = [(L < 0) & (U > 0) for L, U in zip(self.L_list, self.U_list)]
        hwc_unstable_masks = [
            (x.permute(1, 2, 0) if x.dim() == 3 else x)  # Flatten all 3D masks.
            for x in hwc_unstable_masks
        ]

        gurobi_L_list = list(gurobi_results["L_list_unstable_only"])
        gurobi_U_list = list(gurobi_results["U_list_unstable_only"])

        first_layer = next(self.model.children())

        if isinstance(first_layer, nn.Conv2d):
            hwc_shape = hwc_unstable_masks[0].shape
            gurobi_L_list[0] = flattened_hwc_to_chw(gurobi_L_list[0], hwc_shape)
            gurobi_U_list[0] = flattened_hwc_to_chw(gurobi_U_list[0], hwc_shape)

        i = 1
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                i += 1
                continue
            if not isinstance(layer, nn.Conv2d):
                continue

            hwc_shape = hwc_unstable_masks[i].shape
            gurobi_L_list[i] = flattened_unstable_hwc_to_chw(
                gurobi_L_list[i],
                hwc_unstable_masks[i].flatten(),
                hwc_shape,
            )
            gurobi_U_list[i] = flattened_unstable_hwc_to_chw(
                gurobi_U_list[i],
                hwc_unstable_masks[i].flatten(),
                hwc_shape,
            )
            i += 1

        return {
            "L_list_unstable_only": gurobi_L_list,
            "U_list_unstable_only": gurobi_U_list,
            "compute_time": gurobi_results["compute_time"],
        }

    def _mask_p_for_unstable_only(self) -> None:
        """Mask `P` and `P_hat` if not done yet, picking only the unstable neurons."""
        for i in range(len(self.P_list)):
            L, U, P, P_hat = (
                self.L_list[i + 1],
                self.U_list[i + 1],
                self.P_list[i],
                self.P_hat_list[i],
            )
            unstable_mask = torch.flatten((L < 0) & (U > 0))
            self.P_list[i] = P[:, unstable_mask] if len(unstable_mask) == P.size(1) else P
            self.P_hat_list[i] = (
                P_hat[:, unstable_mask] if len(unstable_mask) == P_hat.size(1) else P_hat
            )

    def _unflatten_bounds(self, is_hwc: bool) -> None:
        first_node = self.graph_wrapper.first_child

        shape = first_node.unbatched_input_shape
        if is_hwc and len(shape) == 3:
            C, H, W = shape
            shape = (H, W, C)
        self.L_list[0] = self.L_list[0].reshape(shape)
        self.U_list[0] = self.U_list[0].reshape(shape)

        relu_nodes = (node for node in self.graph_wrapper if isinstance(node.module, nn.ReLU))
        for i, relu in enumerate(relu_nodes, start=1):
            shape = relu.unbatched_output_shape
            if is_hwc and len(shape) == 3:
                C, H, W = shape
                shape = (H, W, C)
            self.L_list[i] = self.L_list[i].reshape(shape)
            self.U_list[i] = self.U_list[i].reshape(shape)

    def _convert_hwc_to_chw(self) -> None:
        """Converts the tensor inputs from Height-Width-Channel (HWC) format to
        the Channel-Height-Width (CHW) format that Pytorch expects.

        Warning: Assumes that `height == width` for all CNN inputs.
        """
        if len(self.graph_wrapper.first_child.unbatched_input_shape) == 3:
            self.L_list[0] = self.L_list[0].permute(2, 0, 1)
            self.U_list[0] = self.U_list[0].permute(2, 0, 1)

        relu_nodes = (node for node in self.graph_wrapper if isinstance(node.module, nn.ReLU))
        for i, relu in enumerate(relu_nodes, start=1):
            if len(relu.unbatched_input_shape) != 3:
                continue

            C, H, W = relu.unbatched_input_shape
            hwc_shape = (H, W, C)
            unstable_mask = torch.flatten((self.L_list[i] < 0) & (self.U_list[i] > 0))
            self.L_list[i] = self.L_list[i].permute(2, 0, 1)
            self.U_list[i] = self.U_list[i].permute(2, 0, 1)
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
        def product(s: Sequence[int]) -> int:
            """Gets the product of all elements (eg. `product((1, 2, 3)) == 6`)."""
            return functools.reduce(lambda x, y: x * y, s)

        _relu_nodes = (node for node in self.graph_wrapper if isinstance(node.module, nn.ReLU))
        relu_shapes = [relu.unbatched_input_shape for relu in _relu_nodes]
        num_relu_layers = len(relu_shapes)
        unbatched_output_shape = self.graph_wrapper.last_child.unbatched_output_shape

        # fmt: off
        assert len(unbatched_output_shape) == 1, f"Expected unbatched output to be 1D, but got {len(unbatched_output_shape)}D."
        num_output_neurons = unbatched_output_shape[0]
        assert 0 <= self.ground_truth_neuron_index < num_output_neurons, f"Expected 0 <= ground_truth_neuron_index < {num_output_neurons}, but got {self.ground_truth_neuron_index} ({num_output_neurons} is the num of neurons in the output layer)."
        assert self.H.size(1) == num_output_neurons, f"Expected H.size(1) == num of output neurons, but got {self.H.size(1)} == {num_output_neurons}."
        assert len(self.L_list) == len(self.U_list) == num_relu_layers + 1, f"Expected len(L_list) == len(U_list) == num of relu layers in `model` + 1 (+ input layer), but got {len(self.L_list)} == {len(self.U_list)} == {num_relu_layers + 1}."
        assert len(self.P_list) == len(self.P_hat_list) == len(self.p_list) == num_relu_layers, f"Expected len(P_list) == len(P_hat_list) == len(p_list) == num of relu layers in `model`, but got {len(self.P_list)} == {len(self.P_hat_list)} == {len(self.p_list)} == {num_relu_layers}."
        # fmt: on

        # Check bounds for each layer.
        bounds_error_msg = "Expected num of elements in L_list[{i}], U_list[{i}] and model's input to all match, but got {self.L_list[i].numel()}, {self.U_list[i].numel()}, {product(shape)}."

        # Input layer.
        assert self.L_list[0].numel() == self.U_list[0].numel() == product(self.input_shape), bounds_error_msg.format(i=0, shape=self.input_shape)  # fmt: skip

        # Intermediate layers.
        for i, relu_shape in enumerate(relu_shapes, start=1):
            L, U, P, P_hat, p = self.L_list[i], self.U_list[i], self.P_list[i - 1], self.P_hat_list[i - 1], self.p_list[i - 1]  # fmt: skip

            assert L.numel() == U.numel() == product(relu_shape), bounds_error_msg.format(i=i, shape=relu_shape)  # fmt: skip
            unstable_mask = (L < 0) & (U > 0)
            num_unstable = int(unstable_mask.sum().item())
            assert P.size(1) == P_hat.size(1) == num_unstable, f"Expected P_list[{i}].size(1) == P_hat_list[{i}].size(1) == num of unstable neurons for ReLU {i}, but got {P.size(1)} == {P_hat.size(1)} == {num_unstable}."  # fmt: skip


def ensure_tensor(array_or_tensor: Union[ndarray, Tensor]) -> Tensor:
    """Converts `array_or_tensor` to a Pytorch Tensor if necessary."""
    return (
        torch.from_numpy(array_or_tensor)
        if isinstance(array_or_tensor, ndarray)
        else array_or_tensor
    )
