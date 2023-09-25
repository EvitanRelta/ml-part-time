import onnx
import onnx2torch
from torch import nn


def load_onnx_model(onnx_file_path: str) -> nn.Module:
    onnx_model = onnx.load(onnx_file_path)
    return onnx2torch.convert(onnx_model)
