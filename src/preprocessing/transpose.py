from typing import Tuple, overload

import torch
from torch import nn

from .class_definitions import Bias, Conv2dFlattenBias, LinearBias, UnaryForward


# fmt: off
@overload
def transpose_layer(layer: nn.Linear) -> Tuple[nn.Linear, Bias]: ...
@overload
def transpose_layer(layer: nn.Conv2d) -> Tuple[nn.ConvTranspose2d, Bias]: ...
# fmt: on
def transpose_layer(layer: nn.Module) -> Tuple[UnaryForward, Bias]:
    """Convert `layer` to a transposed of itself without bias, and return it
    along with a `Bias` module that performs the `V_i^T.b` operation.

    Returns:
        The tranposed layer, and the corresponding `Bias` module.
    """
    if isinstance(layer, nn.Linear):
        return transpose_linear(layer)
    if isinstance(layer, nn.Conv2d):
        return transpose_conv2d(layer)
    raise NotImplementedError()


def transpose_linear(linear: nn.Linear) -> Tuple[nn.Linear, Bias]:
    weight = linear.weight
    bias = linear.bias if linear.bias is not None else torch.zeros((weight.size(0),))

    # Create a new Linear layer with transposed weight and without bias
    transposed_linear = nn.Linear(weight.size(1), weight.size(0), bias=False)
    transposed_linear.weight = nn.Parameter(weight.t().clone().detach(), requires_grad=False)

    return transposed_linear, LinearBias(bias.clone().detach())


def transpose_conv2d(conv2d: nn.Conv2d) -> Tuple[nn.ConvTranspose2d, Bias]:
    weight = conv2d.weight
    bias = conv2d.bias if conv2d.bias is not None else torch.zeros((conv2d.out_channels,))

    # Create a new ConvTranspose2d layer with same parameters and without bias
    transposed_conv2d = nn.ConvTranspose2d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=conv2d.kernel_size,  # type: ignore
        stride=conv2d.stride,  # type: ignore
        padding=conv2d.padding,  # type: ignore
        dilation=conv2d.dilation,  # type: ignore
        groups=conv2d.groups,
        bias=False,
    )
    transposed_conv2d.weight = nn.Parameter(weight.clone().detach(), requires_grad=False)

    return transposed_conv2d, Conv2dFlattenBias(bias.clone().detach())
