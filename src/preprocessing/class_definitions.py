from abc import ABC, abstractmethod
from typing import Protocol, Tuple

import torch
from torch import Tensor, nn
from typing_extensions import override


class UnaryForward(Protocol):
    """Protocol for a Pytorch module with the forward method type: `forward(Tensor) -> Tensor`."""

    def forward(self, input: Tensor) -> Tensor:
        ...


class ConvTranspose2dFlattenNoBias(nn.Module):
    """Module that unflattens input tensor, performs conv2d-transpose operation
    without bias, then flattens output tensor again.
    """

    def __init__(self, conv2d: nn.Conv2d, conv2d_output_shape: Tuple[int, int, int]):
        """
        Args:
            conv2d (nn.Conv2d): `Conv2d` instance to transpose.
            conv2d_output_shape (Tuple[int, int, int]): Output shape of `conv2d`: `(num_channels, H, W)`.
        """
        super().__init__()
        weight = conv2d.weight

        # Create a new ConvTranspose2d layer with same parameters and without bias
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,  # type: ignore
            stride=conv2d.stride,  # type: ignore
            padding=conv2d.padding,  # type: ignore
            dilation=conv2d.dilation,  # type: ignore
            groups=conv2d.groups,
            bias=False,
        )
        self.transposed_conv2d.weight = nn.Parameter(weight.clone().detach(), requires_grad=False)

        self.unflatten = nn.Unflatten(1, conv2d_output_shape)
        self.flatten = nn.Flatten()

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Shape `(num_batches, N)`.

        Returns:
            Tensor: Shape `(num_batches, M)`.
        """
        x = input
        x = self.unflatten.forward(x)
        x = self.transposed_conv2d.forward(x)
        x = self.flatten.forward(x)
        return x


class Bias(nn.Module, ABC, UnaryForward):
    """Base class for generalising the `V_i^T . b` operation in the objective function."""

    def __init__(self, bias: Tensor, is_batched: bool = True) -> None:
        super().__init__()
        self.bias: Tensor
        self.is_batched = is_batched
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
        return V.sum(dim=(-2, -1)) @ self.bias


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
        if self.is_batched:
            num_batches = V.size(0)
            return V.reshape(num_batches, num_channels, -1).sum(dim=(-1)) @ self.bias
        return V.reshape(num_channels, -1).sum(dim=(-2, -1)) @ self.bias


class InverseBatchNorm2d(nn.Module, UnaryForward):
    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        assert bn.running_mean is not None
        assert bn.running_var is not None
        self.mean: Tensor = bn.running_mean.clone().detach()
        self.var: Tensor = bn.running_var.clone().detach()
        self.weight: Tensor = bn.weight.clone().detach()
        self.eps: float = bn.eps

    def forward(self, V: Tensor) -> Tensor:
        return (
            V
            * self.weight[None, :, None, None]
            * torch.sqrt(self.var[None, :, None, None] + self.eps)
            + self.mean[None, :, None, None]
        )
