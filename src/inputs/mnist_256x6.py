import os

import torch
from torch import nn

from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model
from .save_file_types import GurobiResults, SolverInputsSavedDict
from .utils import set_abs_path_to

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)


model: nn.Module = load_onnx_model(get_abs_path("mnist_256x6.onnx"))
loaded_vars: SolverInputsSavedDict = torch.load(get_abs_path("mnist_256x6.pth"))
solver_inputs = SolverInputs(
    model=model,
    ground_truth_neuron_index=loaded_vars["ground_truth_neuron_index"],
    L_list=loaded_vars["L_list"],
    U_list=loaded_vars["U_list"],
    H=loaded_vars["H"],
    d=loaded_vars["d"],
    P_list=loaded_vars["P_list"],
    P_hat_list=loaded_vars["P_hat_list"],
    p_list=loaded_vars["p_list"],
)
gurobi_results: GurobiResults = torch.load(get_abs_path("mnist_256x6_gurobi_results.pth"))
