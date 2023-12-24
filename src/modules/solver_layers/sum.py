from typing import Tuple

import torch
from torch import Tensor, nn


class Sum_SL(nn.Module):
    def forward(self, *args: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        V_list, V_W_list, accum_sum_list = zip(*args)

        # `V` is undefined when summing the L1 layers' outputs, and won't be used by next layer.
        # Thus return a placeholder zero-tensor as `V`.
        V = torch.zeros(0)

        V_W = torch.stack(V_W_list).sum(dim=0)
        accum_sum = torch.stack(accum_sum_list).sum(dim=0)
        return (V, V_W, accum_sum)
