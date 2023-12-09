import os
from typing import List

import torch
from torch import Tensor, nn

from ..preprocessing.solver_inputs import SolverInputs
from ..utils import load_onnx_model, set_abs_path_to

CURRENT_DIR = os.path.dirname(__file__)
get_abs_path = set_abs_path_to(CURRENT_DIR)

model: nn.Module = load_onnx_model(get_abs_path("toy_example.onnx"))

L_list: List[Tensor] = [
    torch.tensor([-1, -1]).float(),
    torch.tensor([0.8, -2, -2]).float(),
    torch.tensor([0.8, 0, 0]).float(),
]
"""Lower limits for neurons. Each List corresponds to the lower limits for a
network layer (ie. index-0 is the lower limits for each neuron in layer-0, the
input layer)."""

U_list: List[Tensor] = [
    torch.tensor([1, 1]).float(),
    torch.tensor([4.8, 2, 2]).float(),
    torch.tensor([4.8, 2, 2]).float(),
]
"""Upper limits for neurons. Each List corresponds to the upper limits for a
network layer (ie. index-0 is the upper limits for each neuron in layer-0, the
input layer)."""

# constraint Hx(L_list)+d <= 0, w.r.t output neurons
# y1-y2-y3 <= 0, -y2 <= 0, -y3 <= 0, 1-1.25y1 <= 0, y1-2 <= 0, y2-2 <= 0, y3-2 <= 0
H: Tensor = torch.tensor(
    [
        [1, -1, -1],
        [0, -1, 0],
        [0, 0, -1],
        [-1.25, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
).float()
"""`H` matrix in the constraint: `Hx(L_list) + d <= 0`, w.r.t output neurons."""

d: Tensor = torch.tensor([0, 0, 0, 1, -2, -2, -2]).float()
"""`d` vector in the constraint: `Hx(L_list) + d <= 0`, w.r.t output neurons."""


# constraint Pxi + P_hatxi_hat - p_list <= 0, w.r.t intermediate unstable neurons and their respective inputs
# -x7 <= 0, -x8 <= 0, -0.5x4 +x7 -1 <= 0, -0.5x5+x8 -1 <= 0, 2x4+x5-x7-x8 <= 0, -x7-x8-2 <= 0
# xi is [x4, x5], xi_hat is [x7, x8]
P_list: List[Tensor] = [
    torch.zeros((1, 2)).float(),
]
"""`P_list` matrix in the constraint `Pxi + P_hatxi_hat - p_list <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

P_hat_list: List[Tensor] = [
    torch.zeros((1, 2)).float(),
]

"""`P_hat_list` matrix in the constraint `Pxi + P_hatxi_hat - p_list <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

p_list: List[Tensor] = [
    torch.zeros((1,)).float(),
]
"""`p_list` vector in the constraint `Pxi + P_hatxi_hat - p_list <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

ground_truth_neuron_index: int = 0

solver_inputs = SolverInputs(
    model=model,
    ground_truth_neuron_index=ground_truth_neuron_index,
    L_list=L_list,
    U_list=U_list,
    H=H,
    d=d,
    P_list=P_list,
    P_hat_list=P_hat_list,
    p_list=p_list,
)
