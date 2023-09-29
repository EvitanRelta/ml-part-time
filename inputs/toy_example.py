import torch
from torch import Tensor, nn

from utils import load_onnx_model

model: nn.Module = load_onnx_model("./inputs/toy_example.onnx")


L: list[Tensor] = [
    torch.tensor([-1, -1]).float(),
    torch.tensor([0.8, -2, -2]).float(),
    torch.tensor([0.8, 0, 0]).float(),
]
"""Lower limits for neurons. Each list corresponds to the lower limits for a
network layer (ie. index-0 is the lower limits for each neuron in layer-0, the
input layer)."""

U: list[Tensor] = [
    torch.tensor([1, 1]).float(),
    torch.tensor([4.8, 2, 2]).float(),
    torch.tensor([4.8, 2, 2]).float(),
]
"""Upper limits for neurons. Each list corresponds to the upper limits for a
network layer (ie. index-0 is the upper limits for each neuron in layer-0, the
input layer)."""

# constraint Hx(L)+d <= 0, w.r.t output neurons
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
"""`H` matrix in the constraint: `Hx(L) + d <= 0`, w.r.t output neurons."""

d: Tensor = torch.tensor([0, 0, 0, 1, -2, -2, -2]).float()
"""`d` vector in the constraint: `Hx(L) + d <= 0`, w.r.t output neurons."""


# constraint Pxi + P_hatxi_hat - p <= 0, w.r.t intermediate unstable neurons and their respective inputs
# -x7 <= 0, -x8 <= 0, -0.5x4 +x7 -1 <= 0, -0.5x5+x8 -1 <= 0, 2x4+x5-x7-x8 <= 0, -x7-x8-2 <= 0
# xi is [x4, x5], xi_hat is [x7, x8]
P: list[Tensor] = [
    torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, -0.5, 0],
            [0, 0, -0.5],
            [0, 2, 1],
            [0, 0, 0],
        ]
    ).float(),
]
"""`P` matrix in the constraint `Pxi + P_hatxi_hat - p <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

P_hat: list[Tensor] = [
    torch.tensor(
        [
            [0, -1, 0],
            [0, 0, -1],
            [0, 1, 0],
            [0, 0, 1],
            [0, -1, -1],
            [0, -1, -1],
        ]
    ).float(),
]

"""`P_hat` matrix in the constraint `Pxi + P_hatxi_hat - p <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""

p: list[Tensor] = [
    torch.tensor([0, 0, 1, 1, 0, 2]).float(),
]
"""`p` vector in the constraint `Pxi + P_hatxi_hat - p <= 0`, w.r.t
intermediate unstable neurons and their respective inputs."""


# Testing value.
_gamma: Tensor = torch.tensor([3, 1, 4, 1, 5, 9, 2]).float()
_pi: list[Tensor] = [torch.tensor([0.5, 2, 3, 8, 5, 6]).float()]
_alpha: list[Tensor] = [torch.tensor([0.4, 0.6]).float()]
