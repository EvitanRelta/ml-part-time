import os

import torch
from torch import nn

from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model, set_abs_path_to
from .save_file_types import GurobiResults, SolverInputsSavedDict

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
ONNX_MODEL_PATH = get_abs_path("conv_med.onnx")
OTHER_INPUTS_PATH = get_abs_path("conv_med.pth")
GUROBI_RESULTS_PATH = get_abs_path("conv_med_gurobi_results.pth")


model: nn.Module = load_onnx_model(ONNX_MODEL_PATH)
model_wo_norm = nn.Sequential(*list(model.children())[4:])

loaded: SolverInputsSavedDict = torch.load(OTHER_INPUTS_PATH)
solver_inputs = SolverInputs(model_wo_norm, **loaded)

gurobi_results: GurobiResults = torch.load(GUROBI_RESULTS_PATH)
gurobi_results = solver_inputs.convert_gurobi_hwc_to_chw(
    gurobi_results,
    hwc_L_list=loaded["L_list"],
    hwc_U_list=loaded["U_list"],
)
