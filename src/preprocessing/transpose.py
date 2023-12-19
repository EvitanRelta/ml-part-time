import math
from functools import reduce
from typing import Tuple

import torch
from torch import nn

from .class_definitions import (
    Bias,
    Conv2dFlattenBias,
    ConvTranspose2dFlattenNoBias,
    LinearBias,
    UnaryForward,
)


def transpose_layer(layer: nn.Module, layer_out_features: int) -> Tuple[UnaryForward, Bias, int]:
    """Convert `layer` to a transposed of itself without bias, and return it
    along with a `Bias` module that performs the `V_i^T.b` operation, and the
    output features of the tranposed layer.

    Args:
        layer (nn.Module): Layer to transpose.
        layer_out_features (int): Output features of `layer`.

    Returns:
        The tranposed layer, and the corresponding `Bias` module.
    """
    if isinstance(layer, nn.Linear):
        return transpose_linear(layer)
    if isinstance(layer, nn.Conv2d):
        return transpose_conv2d(layer, layer_out_features)
    raise NotImplementedError()


def transpose_linear(linear: nn.Linear) -> Tuple[nn.Linear, Bias, int]:
    weight = linear.weight
    bias = linear.bias if linear.bias is not None else torch.zeros((weight.size(0),))

    # Create a new Linear layer with transposed weight and without bias
    transposed_linear = nn.Linear(
        in_features=linear.out_features,
        out_features=linear.in_features,
        bias=False,
    )
    transposed_linear.weight = nn.Parameter(weight.t().clone().detach(), requires_grad=False)

    return transposed_linear, LinearBias(bias.clone().detach()), linear.in_features


def transpose_conv2d(conv2d: nn.Conv2d, conv2d_total_output: int) -> Tuple[UnaryForward, Bias, int]:
    num_channels = conv2d.out_channels

    # Assume that `height == width` for the CNN input.
    H_W = int(math.sqrt(conv2d_total_output / num_channels))
    conv2d_output_shape = (num_channels, H_W, H_W)

    bias = (
        conv2d.bias.clone().detach()
        if conv2d.bias is not None
        else torch.zeros((conv2d.out_channels,))
    )

    output_shape = compute_conv2d_input_shape(conv2d, conv2d_output_shape)
    output_num_elements = reduce(lambda x, y: x * y, output_shape)
    return (
        ConvTranspose2dFlattenNoBias(conv2d, conv2d_output_shape),
        Conv2dFlattenBias(bias),
        output_num_elements,
    )


def compute_conv2d_input_shape(
    conv2d: nn.Conv2d,
    output_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Calculate the input shape of a Conv2d layer given its configuration and output shape.

    Args:
        conv2d (nn.Conv2d): The 2D CNN layer.
        output_shape (Tuple[int, int, int]): Shape of the output tensor in the form: \
            `(num_channels, height, width)`.

    Returns:
        Tuple[int, int, int]: Input tensor shape in the form: \
            `(num_channels, height, width)`.
    """
    _, out_height, out_width = output_shape
    in_channels = conv2d.in_channels
    kernel_size = conv2d.kernel_size
    stride = conv2d.stride
    padding = conv2d.padding
    assert isinstance(padding, Tuple)

    # Calculating the input height and width
    input_height = (out_height - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
    input_width = (out_width - 1) * stride[1] - 2 * padding[1] + kernel_size[1]

    return (in_channels, input_height, input_width)
