import os

import torch
from torch import nn

from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model
from .save_file_types import GurobiResults, SolverInputsSavedDict
from .utils import set_abs_path_to

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)


model: nn.Module = load_onnx_model(get_abs_path("conv_med.onnx"))
model_wo_norm = nn.Sequential(*list(model.children())[4:])
loaded_vars: SolverInputsSavedDict = torch.load(get_abs_path("conv_med.pth"))
solver_inputs = SolverInputs(
    model=model_wo_norm,
    ground_truth_neuron_index=loaded_vars["ground_truth_neuron_index"],
    L=loaded_vars["L"],
    U=loaded_vars["U"],
    H=loaded_vars["H"],
    d=loaded_vars["d"],
    P=loaded_vars["P"],
    P_hat=loaded_vars["P_hat"],
    p=loaded_vars["p"],
)
gurobi_results: GurobiResults = torch.load(get_abs_path("conv_med_gurobi_results.pth"))
