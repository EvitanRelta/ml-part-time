import os

import torch
from torch import Tensor, nn

from inputs.save_file_types import GurobiResults
from preprocessing.solver_inputs import SolverInputs
from utils import load_onnx_model

CURRENT_DIR = os.path.dirname(__file__)

model: nn.Module = load_onnx_model(os.path.join(CURRENT_DIR, "mnist_256x6.onnx"))

loaded_vars = torch.load(os.path.join(CURRENT_DIR, "mnist_256x6.pth"))

L: list[Tensor] = loaded_vars["L"]
"""Lower limits for neurons. Each list corresponds to the lower limits for a
network layer (ie. index-0 is the lower limits for each neuron in layer-0, the
input layer)."""

U: list[Tensor] = loaded_vars["U"]
"""Upper limits for neurons. Each list corresponds to the upper limits for a
network layer (ie. index-0 is the upper limits for each neuron in layer-0, the
input layer)."""


# constraint Hx(L)+d <= 0, w.r.t output neurons
# y1-y2-y3 <= 0, -y2 <= 0, -y3 <= 0, 1-1.25y1 <= 0, y1-2 <= 0, y2-2 <= 0, y3-2 <= 0
H: Tensor = loaded_vars["H"]
"""`H` matrix in the constraint: `Hx(L) + d <= 0`, w.r.t output neurons."""

d: Tensor = loaded_vars["d"]
"""`d` vector in the constraint: `Hx(L) + d <= 0`, w.r.t output neurons."""


# constraint Pxi + P_hatxi_hat - p <= 0, w.r.t intermediate unstable neurons and their respective inputs
# -x7 <= 0, -x8 <= 0, -0.5x4 +x7 -1 <= 0, -0.5x5+x8 -1 <= 0, 2x4+x5-x7-x8 <= 0, -x7-x8-2 <= 0
# xi is [x4, x5], xi_hat is [x7, x8]
P: list[Tensor] = loaded_vars["P"]
"""`P` matrix in the constraint `Pxi + P_hatxi_hat - p <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

P_hat: list[Tensor] = loaded_vars["P_hat"]
"""`P_hat` matrix in the constraint `Pxi + P_hatxi_hat - p <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

p: list[Tensor] = loaded_vars["p"]
"""`p` vector in the constraint `Pxi + P_hatxi_hat - p <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

solver_inputs = SolverInputs(model, 7, L, U, H, d, P, P_hat, p)
gurobi_results: GurobiResults = torch.load(
    os.path.join(CURRENT_DIR, "mnist_256x6_gurobi_results.pth")
)
