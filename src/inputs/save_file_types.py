from typing import List, TypedDict

from torch import Tensor


class GurobiResults(TypedDict):
    L_unstable_only: List[Tensor]
    U_unstable_only: List[Tensor]
    compute_time: float


class SolverInputsSavedDict(TypedDict):
    L: List[Tensor]
    U: List[Tensor]
    H: Tensor
    d: Tensor
    P: List[Tensor]
    P_hat: List[Tensor]
    p: List[Tensor]
    ground_truth_neuron_index: int
