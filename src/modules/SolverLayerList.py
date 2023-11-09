from typing import Iterator, List, Literal, overload

from torch import Tensor, nn

from src.preprocessing.solver_inputs import SolverInputs

from ..preprocessing import preprocessing_utils
from ..preprocessing.transpose import UnaryForward, transpose_model
from .solver_layers.base_class import SolverLayer
from .solver_layers.SolverInput import SolverInput
from .solver_layers.SolverIntermediate import SolverIntermediate
from .solver_layers.SolverOutput import SolverOutput


class SolverLayerList(nn.ModuleList):
    def __init__(self, inputs: SolverInputs):
        preprocessing_utils.freeze_model(inputs.model)
        transposed_layers, b_list = transpose_model(inputs.model)
        (
            stably_act_masks,
            stably_deact_masks,
            unstable_masks,
        ) = preprocessing_utils.get_masks(inputs.L_list, inputs.U_list)

        # Initially set to solve for input layer.
        C_list, self.solve_coords = preprocessing_utils.get_C_for_layer(0, unstable_masks)

        self.layers: List[SolverLayer] = SolverLayerList._split_vars_per_layer(
            inputs,
            transposed_layers,
            b_list,
            stably_act_masks,
            stably_deact_masks,
            unstable_masks,
            C_list,
        )
        super().__init__(self.layers)
        self.d: Tensor
        self.register_buffer("d", inputs.d)

    def solve_for_layer(self, layer_index: int) -> None:
        C_list, self.solve_coords = preprocessing_utils.get_C_for_layer(
            layer_index, self.unstable_masks
        )
        for i in range(len(self.layers)):
            self.layers[i].set_C_and_reset(C_list[i])

    @staticmethod
    def _split_vars_per_layer(
        inputs: SolverInputs,
        transposed_layers: List[UnaryForward],
        b_list: List[Tensor],
        stably_act_masks: List[Tensor],
        stably_deact_masks: List[Tensor],
        unstable_masks: List[Tensor],
        C_list: List[Tensor],
    ) -> List[SolverLayer]:
        layers: List[SolverLayer] = []

        # First-layer inputs.
        layers.append(
            SolverInput(
                L=inputs.L_list[0],
                U=inputs.U_list[0],
                stably_act_mask=stably_act_masks[0],
                stably_deact_mask=stably_deact_masks[0],
                unstable_mask=unstable_masks[0],
                C=C_list[0],
            )
        )

        # Intermediate-layer inputs.
        for i in range(1, len(b_list)):
            layers.append(
                SolverIntermediate(
                    L=inputs.L_list[i],
                    U=inputs.U_list[i],
                    stably_act_mask=stably_act_masks[i],
                    stably_deact_mask=stably_deact_masks[i],
                    unstable_mask=unstable_masks[i],
                    C=C_list[i],
                    transposed_layer=transposed_layers[i - 1],
                    b=b_list[i - 1],
                    transposed_layer_next=transposed_layers[i],
                    P=inputs.P_list[i - 1],
                    P_hat=inputs.P_hat_list[i - 1],
                    p=inputs.p_list[i - 1],
                )
            )

        # Last-layer inputs.
        layers.append(
            SolverOutput(
                L=inputs.L_list[-1],
                U=inputs.U_list[-1],
                stably_act_mask=stably_act_masks[-1],
                stably_deact_mask=stably_deact_masks[-1],
                unstable_mask=unstable_masks[-1],
                C=C_list[-1],
                transposed_layer=transposed_layers[-1],
                b=b_list[-1],
                H=inputs.H,
            )
        )

        return layers

    def __iter__(self) -> Iterator[SolverLayer]:
        return super().__iter__()  # type: ignore

    # fmt: off
    @overload
    def __getitem__(self, i: Literal[0]) -> SolverInput: ...
    @overload
    def __getitem__(self, i: Literal[-1]) -> SolverOutput: ...
    @overload
    def __getitem__(self, i: int) -> SolverIntermediate: ...
    # fmt: on
    def __getitem__(self, i: int) -> SolverLayer:
        return super().__getitem__(i)  # type: ignore

    @property
    def L_list(self) -> List[Tensor]:
        return [x.L for x in self]

    @property
    def U_list(self) -> List[Tensor]:
        return [x.U for x in self]

    @property
    def H(self) -> Tensor:
        return self[-1].H

    @property
    def b_list(self) -> List[Tensor]:
        return [self[i].b for i in range(1, len(self))]

    @property
    def stably_act_masks(self) -> List[Tensor]:
        return [x.stably_act_mask for x in self]

    @property
    def stably_deact_masks(self) -> List[Tensor]:
        return [x.stably_deact_mask for x in self]

    @property
    def unstable_masks(self) -> List[Tensor]:
        return [x.unstable_mask for x in self]

    @property
    def C_list(self) -> List[Tensor]:
        return [x.C for x in self]
