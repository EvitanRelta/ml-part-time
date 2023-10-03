from dataclasses import dataclass
from typing import Literal, Sequence, overload

import torch
from torch import Tensor, nn

from inputs.toy_example import P_hat


@dataclass
class LayerInputs:
    num_neurons: int
    L_i: Tensor
    U_i: Tensor


@dataclass
class InputLayerInputs(LayerInputs):
    ...


@dataclass
class OutputLayerInputs(LayerInputs):
    W_i: Tensor
    b_i: Tensor
    H: Tensor
    initial_gamma: Tensor | None = None


@dataclass
class IntermediateLayerInputs(LayerInputs):
    W_i: Tensor
    b_i: Tensor
    W_next: Tensor
    P_i: Tensor
    P_hat_i: Tensor
    p_i: Tensor
    initial_pi_i: Tensor | None = None
    initial_alpha_i: Tensor | None = None


@dataclass
class SolverInputs(Sequence):
    model: nn.Module
    L: list[Tensor]
    U: list[Tensor]
    H: Tensor
    d: Tensor
    P: list[Tensor]
    P_hat: list[Tensor]
    p: list[Tensor]
    initial_gamma: Tensor | None = None
    initial_pi: list[Tensor] | None = None
    initial_alpha: list[Tensor] | None = None

    def __post_init__(self) -> None:
        cls = self.__class__
        self.num_layers, self.W, self.b = cls.decompose_model(self.model)

    def _validate_inputs(self) -> None:
        assert len(self.P) == len(self.P_hat) == len(self.p)
        for i in range(len(self.P)):
            assert self.P[i].dim() == self.P_hat[i].dim() == 2
            assert self.p[i].dim() == 1
            assert self.P[i].shape == self.P_hat[i].shape
            assert self.p[i].size(0) == self.P[i].size(0)

    def __len__(self):
        return self.num_layers + 1

    @overload
    def __getitem__(self, i: Literal[-1]) -> OutputLayerInputs:
        ...

    @overload
    def __getitem__(self, i: Literal[0]) -> InputLayerInputs:
        ...

    @overload
    def __getitem__(self, i: int) -> IntermediateLayerInputs:
        ...

    def __getitem__(self, i: int) -> LayerInputs:
        if i < -1 or i >= len(self):
            raise IndexError(f"Expected an index i, where -1 <= i < {len(self)}, but got {i}.")

        is_input_layer: bool = i == 0
        if is_input_layer:
            return InputLayerInputs(num_neurons=self.W[1].size(1), L_i=self.L[i], U_i=self.U[i])

        is_output_layer: bool = i == -1 or i == self.num_layers
        if is_output_layer:
            return OutputLayerInputs(
                num_neurons=self.W[i].size(0),
                L_i=self.L[i],
                U_i=self.U[i],
                W_i=self.W[i],
                b_i=self.b[i],
                H=self.H,
                initial_gamma=self.initial_gamma.clone().detach()
                if self.initial_gamma is not None
                else None,
            )

        return IntermediateLayerInputs(
            num_neurons=self.W[i].size(0),
            L_i=self.L[i],
            U_i=self.U[i],
            W_i=self.W[i],
            b_i=self.b[i],
            W_next=self.W[i + 1],
            P_i=self.P[i - 1],
            P_hat_i=P_hat[i - 1],
            p_i=self.p[i - 1],
            initial_pi_i=self.initial_pi[i - 1].clone().detach()
            if self.initial_pi is not None
            else None,
            initial_alpha_i=self.initial_alpha[i - 1].clone().detach()
            if self.initial_alpha is not None
            else None,
        )

    @staticmethod
    def decompose_model(
        model: nn.Module,
    ) -> tuple[int, list[Tensor], list[Tensor]]:
        # Freeze model's layers.
        for param in model.parameters():
            param.requires_grad = False

        linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
        num_layers: int = len(linear_layers)

        W: list[Tensor] = [torch.empty(0)] + [
            layer.weight.clone().detach() for layer in linear_layers
        ]
        b: list[Tensor] = [torch.empty(0)] + [
            layer.bias.clone().detach() for layer in linear_layers
        ]
        return num_layers, W, b
