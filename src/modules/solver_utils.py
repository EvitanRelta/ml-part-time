import torch
from torch import Tensor


def bracket_plus(X: Tensor) -> Tensor:
    return torch.clamp(X, min=0)


def bracket_minus(X: Tensor) -> Tensor:
    return -torch.clamp(X, max=0)
