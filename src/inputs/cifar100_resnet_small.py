import os
from typing import List, Tuple, cast

import torch
from torch import Tensor, nn

from ..inputs.save_file_types import SolverInputsSavedDict
from ..preprocessing.graph_module_wrapper import GraphModuleWrapper
from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model, set_abs_path_to
from .save_file_types import SolverInputsSavedDict

# from .save_file_types import GurobiResults

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
ONNX_MODEL_PATH = get_abs_path("data/cifar100_resnet_small.onnx")
# OTHER_INPUTS_PATH = get_abs_path("data/conv_med_img67.pt")
# GUROBI_RESULTS_PATH = get_abs_path("data/conv_med_img67_gurobi_results.pt")

model, input_shape = load_onnx_model(ONNX_MODEL_PATH, return_input_shape=True)


def generate_random_bounds(shape: Tuple[int, ...]) -> Tuple[Tensor, Tensor]:
    tensor1 = torch.rand(shape, dtype=torch.float) - 0.5
    tensor2 = torch.rand(shape, dtype=torch.float) - 0.5
    L = torch.min(tensor1, tensor2).flatten()
    U = torch.max(tensor1, tensor2).flatten()
    return L, U


def generate_random_p_tensors(L: Tensor, U: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    NUM_CONSTRAINTS = 2
    unstable_mask = (L < 0) & (U > 0)
    num_unstable = cast(int, unstable_mask.sum().item())

    P = torch.rand((NUM_CONSTRAINTS, num_unstable), dtype=torch.float) - 0.5
    P_hat = torch.rand((NUM_CONSTRAINTS, num_unstable), dtype=torch.float) - 0.5
    p = torch.rand((NUM_CONSTRAINTS,), dtype=torch.float) - 0.5
    return P, P_hat, p


def generate_random_H_d(num_output_neurons: int) -> Tuple[Tensor, Tensor]:
    NUM_CONSTRAINTS = 3
    H = torch.rand((NUM_CONSTRAINTS, num_output_neurons)) - 0.5
    d = torch.rand((NUM_CONSTRAINTS,)) - 0.5
    return H, d


L_list: List[Tensor] = []
U_list: List[Tensor] = []
P_list: List[Tensor] = []
P_hat_list: List[Tensor] = []
p_list: List[Tensor] = []
graph_wrapper = GraphModuleWrapper(model, input_shape)
L_0, U_0 = generate_random_bounds(graph_wrapper.first_child.unbatched_input_shape)
L_list.append(L_0)
U_list.append(U_0)

for relu in (x for x in graph_wrapper if isinstance(x.module, nn.ReLU)):
    L, U = generate_random_bounds(relu.unbatched_input_shape)
    P, P_hat, p = generate_random_p_tensors(L, U)
    L_list.append(L)
    U_list.append(U)
    P_list.append(P)
    P_hat_list.append(P_hat)
    p_list.append(p)

num_output_shape: int = 1
for dim in graph_wrapper.last_child.output_shape:
    num_output_shape *= dim
H, d = generate_random_H_d(num_output_shape)

loaded: SolverInputsSavedDict = {
    "L_list": L_list,
    "U_list": U_list,
    "H": H,
    "d": d,
    "P_list": P_list,
    "P_hat_list": P_hat_list,
    "p_list": p_list,
    "ground_truth_neuron_index": 0,
    "is_hwc": False,
}
solver_inputs = SolverInputs(model, input_shape, **loaded)

# gurobi_results: GurobiResults = torch.load(GUROBI_RESULTS_PATH)
# gurobi_results = solver_inputs.convert_gurobi_hwc_to_chw(gurobi_results)
