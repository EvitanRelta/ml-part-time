from typing import Tuple

import torch.nn.functional as F
from torch import Tensor
from typing_extensions import override

from .base_class import Base_SL


class Input_SL(Base_SL):
    """The solver layer for the model's "input layer". This layer is the LAST
    to evaluate, as the computation propagates from output-layer to
    intermediate-layers to input-layer.
    """

    @override
    def __init__(
        self,
        L: Tensor,
        U: Tensor,
        C: Tensor,
    ) -> None:
        super().__init__(L, U, C)

    def forward(self, V_W_1: Tensor, accum_sum: Tensor) -> Tuple[Tensor, Tensor]:
        L, U, C = self.L, self.U, self.C

        theta: Tensor = C - V_W_1
        max_objective = (
            accum_sum
            + (F.relu(theta.flatten(1)) @ L.flatten())
            - (F.relu(-theta.flatten(1)) @ U.flatten())
        )
        return max_objective, theta.detach()

    @override
    def clamp_parameters(self) -> None:
        pass
