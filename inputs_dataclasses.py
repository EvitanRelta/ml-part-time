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

    def __post_init__(self):
        self.stably_act_mask: Tensor = self.L_i >= 0
        self.stably_deact_mask: Tensor = self.U_i <= 0
        self.unstable_mask: Tensor = (self.L_i < 0) & (self.U_i > 0)
        assert torch.all((self.stably_act_mask + self.stably_deact_mask + self.unstable_mask) == 1)

        self.num_unstable: int = int(self.unstable_mask.sum().item())


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
        self._validate_inputs()
        self._layer_inputs: list[LayerInputs] = self._get_inputs_per_layer()

    def _validate_inputs(self) -> None:
        assert len(self.P) == len(self.P_hat) == len(self.p)
        for i in range(len(self.P)):
            assert self.P[i].dim() == self.P_hat[i].dim() == 2
            assert self.p[i].dim() == 1
            assert self.P[i].shape == self.P_hat[i].shape
            assert self.p[i].size(0) == self.P[i].size(0)

    def _get_inputs_per_layer(self) -> list[LayerInputs]:
        layer_inputs_list: list[LayerInputs] = []

        # First-layer inputs.
        layer_inputs_list.append(
            InputLayerInputs(num_neurons=self.W[1].size(1), L_i=self.L[0], U_i=self.U[0])
        )

        # Intermediate-layer inputs.
        for i in range(1, self.num_layers):
            layer_inputs_list.append(
                IntermediateLayerInputs(
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
            )

        # Last-layer inputs.
        layer_inputs_list.append(
            OutputLayerInputs(
                num_neurons=self.W[-1].size(0),
                L_i=self.L[-1],
                U_i=self.U[-1],
                W_i=self.W[-1],
                b_i=self.b[-1],
                H=self.H,
                initial_gamma=self.initial_gamma.clone().detach()
                if self.initial_gamma is not None
                else None,
            )
        )

        return layer_inputs_list

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
        return self._layer_inputs[i]

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
