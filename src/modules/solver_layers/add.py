from typing import Tuple

import torch
from torch import Tensor, nn


class Add_SL(nn.Module):
    def forward(self, tuple_args: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        V_next, V_W_next, accum_sum = tuple_args
        return V_next, torch.zeros_like(V_W_next), torch.zeros_like(accum_sum)
