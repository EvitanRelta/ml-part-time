from typing import TypedDict

from torch import Tensor


class GurobiResults(TypedDict):
    L_unstable_only: list[Tensor]
    U_unstable_only: list[Tensor]
    compute_time: float
