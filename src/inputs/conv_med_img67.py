import os

import torch

from ..preprocessing.preprocessing_utils import remove_first_n_modules
from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model, set_abs_path_to
from .save_file_types import GurobiResults, SolverInputsSavedDict

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
ONNX_MODEL_PATH = get_abs_path("conv_med.onnx")
OTHER_INPUTS_PATH = get_abs_path("conv_med_img67.pth")
GUROBI_RESULTS_PATH = get_abs_path("conv_med_img67_gurobi_results.pth")


model = load_onnx_model(ONNX_MODEL_PATH)
model = remove_first_n_modules(model, 4)  # Remove norm layers.

loaded: SolverInputsSavedDict = torch.load(OTHER_INPUTS_PATH)
solver_inputs = SolverInputs(model, **loaded)

gurobi_results: GurobiResults = torch.load(GUROBI_RESULTS_PATH)
gurobi_results = solver_inputs.convert_gurobi_hwc_to_chw(
    gurobi_results,
    hwc_L_list=loaded["L_list"],
    hwc_U_list=loaded["U_list"],
)
