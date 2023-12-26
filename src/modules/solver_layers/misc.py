from typing import Tuple

import torch
from torch import Tensor, nn

from ...preprocessing.class_definitions import UnaryForward


class Misc_SL(nn.Module):
    def __init__(self, transposed_layer: UnaryForward) -> None:
        super().__init__()
        self.transposed_layer = transposed_layer

    def forward(self, tuple_args: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        V_next, V_W_next, accum_sum = tuple_args

        return (
            V_next,
            self.transposed_layer.forward(V_W_next),
            torch.zeros_like(accum_sum),
        )
