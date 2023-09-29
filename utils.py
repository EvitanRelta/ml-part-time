import onnx
import onnx2torch
import torch
from torch import Tensor, nn


def load_onnx_model(onnx_file_path: str) -> nn.Module:
    onnx_model = onnx.load(onnx_file_path)
    return onnx2torch.convert(onnx_model)


def bracket_plus(X: Tensor) -> Tensor:
    return torch.clamp(X, min=0)


def bracket_minus(X: Tensor) -> Tensor:
    return -torch.clamp(X, max=0)
