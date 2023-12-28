import os

import torch

from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model, set_abs_path_to
from .save_file_types import GurobiResults, SolverInputsSavedDict

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
ONNX_MODEL_PATH = get_abs_path("data/mnist_256x6.onnx")
OTHER_INPUTS_PATH = get_abs_path("data/mnist_256x6.pt")
GUROBI_RESULTS_PATH = get_abs_path("data/mnist_256x6_gurobi_results.pt")

model, input_shape = load_onnx_model(ONNX_MODEL_PATH, return_input_shape=True)
loaded: SolverInputsSavedDict = torch.load(OTHER_INPUTS_PATH)
loaded["L_list"] = loaded["L_list"][:-1]
loaded["U_list"] = loaded["U_list"][:-1]
solver_inputs = SolverInputs(model, input_shape, **loaded)

gurobi_results: GurobiResults = torch.load(GUROBI_RESULTS_PATH)
