from torch import Tensor
from typing_extensions import override

from .base_class import SolverLayer


class SolverInput(SolverLayer):
    @override
    def __init__(
        self,
        L: Tensor,
        U: Tensor,
        stably_act_mask: Tensor,
        stably_deact_mask: Tensor,
        unstable_mask: Tensor,
        C: Tensor,
    ) -> None:
        super().__init__(L, U, stably_act_mask, stably_deact_mask, unstable_mask, C)

    # fmt: off
    @override
    def forward(self, V_next: Tensor) -> Tensor: raise NotImplementedError()

    @override
    def clamp_parameters(self) -> None: ...

    @override
    def get_obj_sum(self) -> Tensor: raise NotImplementedError()
    # fmt: on
