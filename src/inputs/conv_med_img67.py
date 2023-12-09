import os
from typing import List

import torch
from torch import nn

from ..preprocessing.linearize_conv2d import (
    compute_conv2d_output_shape,
    linearize_conv2d,
)
from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model, set_abs_path_to
from .save_file_types import GurobiResults, SolverInputsSavedDict

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)


model: nn.Module = load_onnx_model(get_abs_path("conv_med.onnx"))
model_wo_norm = nn.Sequential(*list(model.children())[4:])

# Linearize the model (w/o normalization/flatten layers).
temp: List[nn.Module] = []
input_shape = (3, 32, 32)
for layer in model.children():
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.ReLU):
        temp.append(layer)
    elif isinstance(layer, nn.Conv2d):
        linearized_conv2d = linearize_conv2d(layer, input_shape)
        temp.append(linearized_conv2d)
        input_shape = compute_conv2d_output_shape(layer, input_shape)
model_linearized = nn.Sequential(*temp)

loaded_vars: SolverInputsSavedDict = torch.load(get_abs_path("conv_med_img67.pth"))
solver_inputs = SolverInputs(
    model=model_linearized,
    ground_truth_neuron_index=loaded_vars["ground_truth_neuron_index"],
    L_list=loaded_vars["L_list"],
    U_list=loaded_vars["U_list"],
    H=loaded_vars["H"],
    d=loaded_vars["d"],
    P_list=loaded_vars["P_list"],
    P_hat_list=loaded_vars["P_hat_list"],
    p_list=loaded_vars["p_list"],
)
gurobi_results: GurobiResults = torch.load(get_abs_path("conv_med_img67_gurobi_results.pth"))
