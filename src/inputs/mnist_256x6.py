import os

import torch

from ..preprocessing.solver_inputs import SolverInputs
from ..utils import set_abs_path_to
from .save_file_types import GurobiResults

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)
ONNX_MODEL_PATH = get_abs_path("mnist_256x6.onnx")
OTHER_INPUTS_PATH = get_abs_path("mnist_256x6.pth")
GUROBI_RESULTS_PATH = get_abs_path("mnist_256x6_gurobi_results.pth")

solver_inputs = SolverInputs.load(ONNX_MODEL_PATH, OTHER_INPUTS_PATH)
gurobi_results: GurobiResults = torch.load(GUROBI_RESULTS_PATH)
