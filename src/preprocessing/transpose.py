from typing import List, Tuple, overload

import torch
from torch import Tensor, nn


class UnaryForwardModule(nn.Module):
    def forward(self, X: Tensor) -> Tensor:
        ...


def transpose_model(model: nn.Module) -> Tuple[List[UnaryForwardModule], List[Tensor]]:
    layers = [
        layer
        for layer in model.children()
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)
    ]

    tranposed_layers: List[nn.Module] = []
    biases: List[Tensor] = []
    for l in layers:
        tranposed_layer, b = transpose_layer(l)
        tranposed_layers.append(tranposed_layer)
        biases.append(b)

    return tranposed_layers, biases  # type: ignore


# fmt: off
@overload
def transpose_layer(layer: nn.Linear) -> Tuple[nn.Linear, Tensor]: ...
@overload
def transpose_layer(layer: nn.Conv2d) -> Tuple[nn.ConvTranspose2d, Tensor]: ...
# fmt: on
def transpose_layer(layer: nn.Module) -> Tuple[nn.Module, Tensor]:
    """Convert `layer` to a transposed of itself without bias, and return it
    along with the original bias tensor.

    Returns:
        The tranposed layer, and the original bias tensor.
    """
    if isinstance(layer, nn.Linear):
        return transpose_linear(layer)
    if isinstance(layer, nn.Conv2d):
        return transpose_conv2d(layer)
    raise NotImplementedError()


def transpose_linear(linear: nn.Linear) -> Tuple[nn.Linear, Tensor]:
    """Convert `linear` to a transposed of itself without bias, and return it
    along with the original bias tensor.

    Args:
        linear (nn.Linear): The linear layer to transpose.

    Returns:
        Tuple[nn.Linear, Tensor]: The tranposed linear layer, and the original \
            bias tensor.
    """
    weight = linear.weight
    bias = linear.bias if linear.bias is not None else torch.zeros((weight.size(0),))

    # Create a new Linear layer with transposed weight and without bias
    transposed_linear = nn.Linear(weight.size(1), weight.size(0), bias=False)
    transposed_linear.weight = nn.Parameter(weight.t().clone().detach(), requires_grad=False)

    return transposed_linear, bias.clone().detach()


def transpose_conv2d(conv2d: nn.Conv2d) -> Tuple[nn.ConvTranspose2d, Tensor]:
    """Convert `conv2d` to a transposed of itself without bias, and return it
    along with the original bias tensor.

    Args:
        conv2d (nn.Conv2d): The 2D CNN layer to transpose.

    Returns:
        Tuple[nn.ConvTranspose2d, Tensor]: The tranposed linear layer, and the \
            original bias tensor.
    """
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

    return transposed_conv2d, bias.clone().detach()
