from typing import TypedDict

from torch import Tensor


class GurobiResults(TypedDict):
    L_unstable_only: list[Tensor]
    U_unstable_only: list[Tensor]
    compute_time: float


class SolverInputsSavedDict(TypedDict):
    L: list[Tensor]
    U: list[Tensor]
    H: Tensor
    d: Tensor
    P: list[Tensor]
    P_hat: list[Tensor]
    p: list[Tensor]
