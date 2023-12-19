import torch
from torch import Tensor


def bracket_plus(X: Tensor) -> Tensor:
    """The [X]+ operation. Clamping all elements to be >= 0."""
    return torch.clamp(X, min=0)


def bracket_minus(X: Tensor) -> Tensor:
    """The [X]- operation. Clamping all elements to be <= 0, then negating the
    value.
    """
    return -torch.clamp(X, max=0)
