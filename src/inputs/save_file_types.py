from typing import List, TypedDict

from torch import Tensor
from typing_extensions import NotRequired


class GurobiResults(TypedDict):
    L_list_unstable_only: List[Tensor]
    U_list_unstable_only: List[Tensor]
    compute_time: float


class SolverInputsSavedDict(TypedDict):
    L_list: List[Tensor]
    U_list: List[Tensor]
    H: Tensor
    d: Tensor
    P_list: List[Tensor]
    P_hat_list: List[Tensor]
    p_list: List[Tensor]
    ground_truth_neuron_index: int
    is_hwc: NotRequired[bool]
