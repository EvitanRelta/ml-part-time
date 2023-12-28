from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class Solvable_SL(ABC, nn.Module):
    """Abstract base class for all solver layers that are "solvable"
    (ie. input layer + ReLU layers).

    Contains the common constructor-parameters/methods/properties.
    """

    def __init__(
        self,
        L: Tensor,
        U: Tensor,
        C: Tensor,
    ) -> None:
        super().__init__()
        stably_act_mask = L >= 0
        stably_deact_mask = U <= 0
        unstable_mask = (L < 0) & (U > 0)
        assert torch.all((stably_act_mask + stably_deact_mask + unstable_mask) == 1)

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

    def set_C_and_reset_parameters(self, C: Tensor) -> None:
        """Set `C` tensor and reset learnable parameters."""
        self.register_buffer("C", C)

    @property
    def num_batches(self) -> int:
        """The number of batches this layer is set to solve for."""
        return self.C.size(0)

    @property
    def layer_shape(self) -> torch.Size:
        """The number of neurons this layer has."""
        return self.L.shape

    @property
    def num_unstable(self) -> int:
        """The number of unstable neurons this layer has."""
        return int(self.unstable_mask.sum().item())

    @abstractmethod
    def clamp_parameters(self) -> None:
        """Clamps all learnable parameters to their values' domains.

        Specifically:
        - `gamma >= 0`
        - `pi >= 0`
        - `0 <= alpha <= 1`
        """
        ...
