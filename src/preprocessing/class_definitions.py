from abc import ABC, abstractmethod
from typing import Protocol

from torch import Tensor, nn
from typing_extensions import override


class UnaryForward(Protocol):
    """Protocol for a Pytorch module with the forward method type: `forward(Tensor) -> Tensor`."""

    def forward(self, input: Tensor) -> Tensor:
        ...


class Bias(nn.Module, ABC, UnaryForward):
    """Base class for generalising the `V_i^T . b` operation in the objective function."""

    def __init__(self, bias: Tensor) -> None:
        super().__init__()
        self.bias: Tensor
        self.register_buffer("bias", bias)

    @abstractmethod
    def forward(self, V: Tensor) -> Tensor:
        ...


class LinearBias(Bias):
    @override
    def forward(self, V: Tensor) -> Tensor:
        """Given a bias of shape `(N,)`, apply the bias to `V`.

        Args:
            V (Tensor): Shape `(N,)` or `(num_batches, N)`

        Returns:
            Tensor: Tensor of shape `(1,)` or `(num_batches, 1)`, with the bias applied.
        """
        return V @ self.bias


class Conv2dBias(Bias):
    @override
    def forward(self, V: Tensor) -> Tensor:
        """Given a bias of shape `(num_channels,)`, apply the bias to `V`.

        Args:
            V (Tensor): Shape `(num_channels, H, W)` or `(num_batches, num_channels, H, W)`

        Returns:
            Tensor: Tensor of shape `(1,)` or `(num_batches, 1)`, with the bias applied.
        """
        return V.sum(dim=(-2, -1)) * self.bias


class Conv2dFlattenBias(Bias):
    @override
    def forward(self, V: Tensor) -> Tensor:
        """Given a bias of shape `(num_channels,)`, apply the bias to `V`.

        Args:
            V (Tensor): Shape `(num_channels * H * W)` or `(num_batches, num_channels * H * W)`

        Returns:
            Tensor: Tensor of shape `(1,)` or `(num_batches, 1)`, with the bias applied.
        """
        num_channels = self.bias.size(0)
        has_batches = V.dim() > 1
        if has_batches:
            num_batches = V.size(0)
            return V.reshape(num_batches, num_channels, -1).sum(dim=(-2, -1)) * self.bias
        return V.reshape(num_channels, -1).sum(dim=(-2, -1)) * self.bias
