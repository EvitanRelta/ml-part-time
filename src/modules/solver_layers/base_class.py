from abc import ABC, abstractmethod

from torch import Tensor, nn


class LayerVariables(nn.Module):
    def __init__(
        self,
        L: Tensor,
        U: Tensor,
        stably_act_mask: Tensor,
        stably_deact_mask: Tensor,
        unstable_mask: Tensor,
        C: Tensor,
    ) -> None:
        super().__init__()
        self.L: Tensor
        self.U: Tensor
        self.stably_act_mask: Tensor
        self.stably_deact_mask: Tensor
        self.unstable_mask: Tensor
        self.C: Tensor

        self.register_buffer("L", L)
        self.register_buffer("U", U)
        self.register_buffer("stably_act_mask", stably_act_mask)
        self.register_buffer("stably_deact_mask", stably_deact_mask)
        self.register_buffer("unstable_mask", unstable_mask)
        self.register_buffer("C", C)

    def set_C(self, C: Tensor) -> None:
        self.register_buffer("C", C)

    @property
    def num_batches(self) -> int:
        return self.C.size(0)

    @property
    def num_neurons(self) -> int:
        return len(self.L)

    @property
    def num_unstable(self) -> int:
        return int(self.unstable_mask.sum().item())


class SolverLayer(ABC, nn.Module):
    """Abstract base class for all solver layers."""

    def __init__(self, vars: LayerVariables) -> None:
        super().__init__()
        self.vars = vars

    # fmt: off
    @abstractmethod
    def forward(self, V_next: Tensor) -> Tensor: ...

    @abstractmethod
    def clamp_parameters(self) -> None: ...

    @abstractmethod
    def get_obj_sum(self) -> Tensor: ...
    # fmt: on
