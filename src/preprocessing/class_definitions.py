from typing import Protocol

from torch import Tensor


class UnaryForward(Protocol):
    """Protocol for a Pytorch module with the forward method type: `forward(Tensor) -> Tensor`."""

    def forward(self, input: Tensor) -> Tensor:
        ...
