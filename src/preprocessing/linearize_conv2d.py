from typing import Literal, Tuple, Union

import torch
from torch import Tensor, nn


def linearize_conv2d(
    conv2d: nn.Conv2d,
    input_shape: Union[Tuple[int, int, int], torch.Size],
) -> nn.Linear:
    """Convert a Pytorch `Conv2d` layer to a sparse `Linear` layer.

    Args:
        conv2d (nn.Conv2d): The 2D CNN layer.
        input_shape (Union[Tuple[int, int, int], torch.Size]): Shape of the input tensor in the form: \
            `(num_channels, height, width)`.

    Returns:
        nn.Linear: The linearized CNN layer.
    """
    W, b = conv2d_to_matrices(conv2d, input_shape)  # Assume this function is defined

    # Number of input features for the Linear layer
    num_input_features = (
        input_shape[0] * input_shape[1] * input_shape[2]
    )  # channels * height * width

    # Number of output features is determined from the shape of W
    num_output_features = W.shape[0]

    # Create the Linear layer
    linear = nn.Linear(num_input_features, num_output_features)

    # Set the weights and biases
    linear.weight.data = W
    linear.bias.data = b

    return linear


def conv2d_to_matrices(
    conv2d: nn.Conv2d,
    input_shape: Union[Tuple[int, int, int], torch.Size],
) -> Tuple[Tensor, Tensor]:
    """Convert a Pytorch `Conv2d` layer to a linearized weights tensor `W` and a bias tensor `b`.

    Args:
        conv2d (nn.Conv2d): The 2D CNN layer.
        input_shape (Union[Tuple[int, int, int], torch.Size]): Shape of the input tensor in the form: \
            `(num_channels, height, width)`.

    Returns:
        Tuple[Tensor, Tensor]: The linearized weights & biases tensors: `(W, b)`.
    """
    # Extract weights and biases
    weights = conv2d.weight.data
    assert conv2d.bias is not None
    biases = conv2d.bias.data

    # Unpack input shape and convolution parameters
    C_in, H_in, W_in = input_shape
    out_channels, in_channels, kernel_h, kernel_w = weights.shape
    stride, padding, dilation = (
        conv2d.stride,
        conv2d.padding,
        conv2d.dilation,
    )
    assert isinstance(padding, Tuple)

    # Calculate output dimensions
    H_out = (H_in + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

    # Create the weight matrix W and bias vector b
    W_shape = (H_out * W_out * out_channels, C_in * H_in * W_in)
    W = torch.zeros(W_shape, dtype=weights.dtype)

    for i in range(H_out):
        for j in range(W_out):
            for n in range(out_channels):
                # Calculate the receptive field for this position
                for c in range(C_in):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            h_index = i * stride[0] - padding[0] + kh * dilation[0]
                            w_index = j * stride[1] - padding[1] + kw * dilation[1]
                            if 0 <= h_index < H_in and 0 <= w_index < W_in:
                                # Calculate indices in the flattened input and weight tensors
                                input_index = c * H_in * W_in + h_index * W_in + w_index
                                weight_index = n * H_out * W_out + i * W_out + j
                                W[weight_index, input_index] = weights[n, c, kh, kw]

    # Convert biases to the correct shape
    b = biases.repeat_interleave(H_out * W_out)

    return W, b


def compute_conv2d_output_shape(
    conv2d: nn.Conv2d,
    input_shape: Tuple[int, int, int],
    format: Literal["CHW", "HWC"] = "CHW",
) -> Tuple[int, int, int]:
    """Calculate the output shape of a Conv2d layer given its configuration and input shape.

    Args:
        conv2d (nn.Conv2d): The 2D CNN layer.
        input_shape (Tuple[int, int, int]): Shape of the input tensor in the form: \
            `(num_channels, height, width)`.
        format (Literal["CHW", "HWC"], optional): Format of `input_shape` and of \
            the returned output. Defaults to "CHW".

    Returns:
        Tuple[int, int, int]: Output tensor shape in the form: \
            `(num_channels, height, width)` for `format="CHW"`, or \
            `(height, width, num_channels)` for `format="HWC"`.
    """
    assert format in ("CHW", "HWC")

    # Extract parameters from the conv2d layer
    kernel_size = conv2d.kernel_size
    stride = conv2d.stride
    padding = conv2d.padding
    dilation = conv2d.dilation
    assert isinstance(padding, Tuple)

    # Extract input dimensions (ignoring channel size)
    if format == "CHW":
        _, input_height, input_width = input_shape
    else:
        input_height, input_width, _ = input_shape

    # Calculate the output dimensions based on the equation in PyTorch's Conv2d's docs.
    output_height = (
        (input_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]
    ) + 1
    output_width = (
        (input_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]
    ) + 1

    if format == "CHW":
        return (conv2d.out_channels, output_height, output_width)
    return (output_height, output_width, conv2d.out_channels)
