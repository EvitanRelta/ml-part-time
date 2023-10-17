from dataclasses import dataclass
from typing import Literal, Sequence, overload

import torch
from torch import Tensor, nn


@dataclass
class LayerInputs:
    batches: int
    num_neurons: int
    L_i: Tensor
    U_i: Tensor
    stably_act_mask: Tensor
    stably_deact_mask: Tensor
    unstable_mask: Tensor
    num_unstable: int
    C_i: Tensor


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
        self.stably_act_masks, self.stably_deact_masks, self.unstable_masks = cls._get_masks(
            self.L, self.U
        )
        self.num_unstable_list: list[int] = [int(mask.sum().item()) for mask in self.unstable_masks]
        self.total_num_intermediate_unstable: int = sum(self.num_unstable_list[1:-1])
        self.C = cls._get_C(self.unstable_masks, self.total_num_intermediate_unstable)
        self._layer_inputs: list[LayerInputs] = self._get_inputs_per_layer()

    def _validate_inputs(self) -> None:
        assert len(self.P) == len(self.P_hat) == len(self.p)
        for i in range(len(self.P)):
            assert self.P[i].dim() == self.P_hat[i].dim() == 2
            assert self.p[i].dim() == 1
            assert self.P[i].shape == self.P_hat[i].shape
            assert self.p[i].size(0) == self.P[i].size(0)

    @staticmethod
    def _get_masks(
        L: list[Tensor], U: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        num_layers = len(U)
        stably_act_masks: list[Tensor] = [L_i >= 0 for L_i in L]
        stably_deact_masks: list[Tensor] = [U_i <= 0 for U_i in U]
        unstable_masks: list[Tensor] = [(L[i] < 0) & (U[i] > 0) for i in range(num_layers)]
        for i in range(num_layers):
            assert torch.all((stably_act_masks[i] + stably_deact_masks[i] + unstable_masks[i]) == 1)

        return stably_act_masks, stably_deact_masks, unstable_masks

    @staticmethod
    def _get_C(
        unstable_masks: list[Tensor],
        total_num_intermediate_unstable: int,
    ) -> list[Tensor]:
        l = len(unstable_masks) - 1
        C: list[Tensor] = []

        batch_index: int = 0
        for layer in range(len(unstable_masks)):
            unstable_mask: Tensor = unstable_masks[layer]
            num_neurons: int = unstable_mask.size(0)

            if layer == 0 or layer == l:  # Don't solve for 1st/last layer.
                C.append(torch.zeros((total_num_intermediate_unstable * 2, num_neurons)))
                continue

            unstable_indices: Tensor = torch.where(unstable_mask)[0]
            C_i = torch.zeros((total_num_intermediate_unstable * 2, num_neurons))
            for index in unstable_indices:
                C_i[batch_index][index] = 1  # Minimising
                C_i[batch_index + 1][index] = -1  # Maximising
                batch_index += 2
            C.append(C_i)
        return C

    def _get_inputs_per_layer(self) -> list[LayerInputs]:
        layer_inputs_list: list[LayerInputs] = []

        # First-layer inputs.
        layer_inputs_list.append(
            InputLayerInputs(
                batches=self.total_num_intermediate_unstable * 2,
                num_neurons=self.W[1].size(1),
                L_i=self.L[0],
                U_i=self.U[0],
                stably_act_mask=self.stably_act_masks[0],
                stably_deact_mask=self.stably_deact_masks[0],
                unstable_mask=self.unstable_masks[0],
                num_unstable=self.num_unstable_list[0],
                C_i=self.C[0],
            )
        )

        # Intermediate-layer inputs.
        for i in range(1, self.num_layers):
            layer_inputs_list.append(
                IntermediateLayerInputs(
                    batches=self.total_num_intermediate_unstable * 2,
                    num_neurons=self.W[i].size(0),
                    L_i=self.L[i],
                    U_i=self.U[i],
                    stably_act_mask=self.stably_act_masks[i],
                    stably_deact_mask=self.stably_deact_masks[i],
                    unstable_mask=self.unstable_masks[i],
                    num_unstable=self.num_unstable_list[i],
                    C_i=self.C[i],
                    W_i=self.W[i],
                    b_i=self.b[i],
                    W_next=self.W[i + 1],
                    P_i=self.P[i - 1],
                    P_hat_i=self.P_hat[i - 1],
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
                batches=self.total_num_intermediate_unstable * 2,
                num_neurons=self.W[-1].size(0),
                L_i=self.L[-1],
                U_i=self.U[-1],
                stably_act_mask=self.stably_act_masks[-1],
                stably_deact_mask=self.stably_deact_masks[-1],
                unstable_mask=self.unstable_masks[-1],
                num_unstable=self.num_unstable_list[-1],
                C_i=self.C[-1],
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
