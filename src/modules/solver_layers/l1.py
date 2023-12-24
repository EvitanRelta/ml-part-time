from typing import Tuple

from torch import Tensor, nn

from ...preprocessing.class_definitions import Bias, UnaryForward


class L1_SL(nn.Module):
    def __init__(
        self,
        transposed_layer: UnaryForward,
        bias_module: Bias,
    ) -> None:
        super().__init__()
        self.transposed_layer = transposed_layer
        self.bias_module = bias_module

    def forward(
        self,
        V_next: Tensor,
        V_W_next: Tensor,
        accum_sum: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        V = V_next
        return (
            V,
            self.transposed_layer.forward(V),
            accum_sum - self.bias_module.forward(V),
        )
