from collections.abc import Iterator
from typing import Iterator, List, TypeVar, Union

import pytest
from torch import Tensor, nn

from ..modules.solver_layers.base_class import SolverLayer
from ..modules.solver_layers.SolverInput import SolverInput
from ..modules.solver_layers.SolverIntermediate import SolverIntermediate
from ..modules.solver_layers.SolverOutput import SolverOutput
from . import preprocessing_utils
from .solver_inputs import SolverInputs
from .transpose import transpose_layer

T = TypeVar("T")


def get_reversed_iterator(list_or_iterator: Union[List[T], Iterator[T]]) -> Iterator[T]:
    items = list_or_iterator if isinstance(list_or_iterator, list) else list(list_or_iterator)
    items.reverse()
    return iter(items)


def build(inputs: SolverInputs) -> List[SolverLayer]:
    preprocessing_utils.freeze_model(inputs.model)
    (
        stably_act_masks,
        stably_deact_masks,
        unstable_masks,
    ) = preprocessing_utils.get_masks(inputs.L_list, inputs.U_list)

    # Initially set to solve for input layer.
    C_list, solve_coords = preprocessing_utils.get_C_for_layer(0, unstable_masks)

    layer_gen = get_reversed_iterator(inputs.model.children())
    L_gen = get_reversed_iterator(inputs.L_list)
    U_gen = get_reversed_iterator(inputs.U_list)
    P_gen = get_reversed_iterator(inputs.P_list)
    P_hat_gen = get_reversed_iterator(inputs.P_hat_list)
    p_gen = get_reversed_iterator(inputs.p_list)
    stably_act_mask_gen = get_reversed_iterator(stably_act_masks)
    stably_deact_mask_gen = get_reversed_iterator(stably_deact_masks)
    unstable_mask_gen = get_reversed_iterator(unstable_masks)
    C_gen = get_reversed_iterator(C_list)

    last_layer = next(layer_gen)
    assert isinstance(last_layer, nn.Linear)
    transposed_layer, b = transpose_layer(last_layer)

    output_layer = SolverOutput(
        L=next(L_gen),
        U=next(U_gen),
        stably_act_mask=next(stably_act_mask_gen),
        stably_deact_mask=next(stably_deact_mask_gen),
        unstable_mask=next(unstable_mask_gen),
        C=next(C_gen),
        transposed_layer=transposed_layer,
        b=b,
        H=inputs.H,
        d=inputs.d,
    )
    solver_layers: List[SolverLayer] = [output_layer]

    prev_layer: Union[SolverOutput, SolverIntermediate] = output_layer

    while True:
        try:
            intermediate_layer = build_intermediate_layer(
                layer_gen=layer_gen,
                L_gen=L_gen,
                U_gen=U_gen,
                P_gen=P_gen,
                P_hat_gen=P_hat_gen,
                p_gen=p_gen,
                stably_act_mask_gen=stably_act_mask_gen,
                stably_deact_mask_gen=stably_deact_mask_gen,
                unstable_mask_gen=unstable_mask_gen,
                C_gen=C_gen,
                prev_layer=prev_layer,
            )
            solver_layers.append(intermediate_layer)
            prev_layer = intermediate_layer
        except StopIteration:
            break

    solver_layers.append(
        SolverInput(
            L=next(L_gen),
            U=next(U_gen),
            stably_act_mask=next(stably_act_mask_gen),
            stably_deact_mask=next(stably_deact_mask_gen),
            unstable_mask=next(unstable_mask_gen),
            C=next(C_gen),
            transposed_layer=prev_layer.transposed_layer,
        )
    )

    # Assert that all generators are depleted.
    for gen in [layer_gen, L_gen, U_gen, P_gen, P_hat_gen, p_gen, stably_act_mask_gen, stably_deact_mask_gen, unstable_mask_gen, C_gen]:  # fmt: skip
        with pytest.raises(StopIteration):
            next(gen)

    solver_layers.reverse()
    return solver_layers


def build_intermediate_layer(
    layer_gen: Iterator[nn.Module],
    L_gen: Iterator[Tensor],
    U_gen: Iterator[Tensor],
    P_gen: Iterator[Tensor],
    P_hat_gen: Iterator[Tensor],
    p_gen: Iterator[Tensor],
    stably_act_mask_gen: Iterator[Tensor],
    stably_deact_mask_gen: Iterator[Tensor],
    unstable_mask_gen: Iterator[Tensor],
    C_gen: Iterator[Tensor],
    prev_layer: Union[SolverIntermediate, SolverOutput],
) -> SolverIntermediate:
    layer = next(layer_gen)
    while not isinstance(layer, nn.Linear):
        layer = next(layer_gen)

    transposed_layer, b = transpose_layer(layer)
    return SolverIntermediate(
        L=next(L_gen),
        U=next(U_gen),
        stably_act_mask=next(stably_act_mask_gen),
        stably_deact_mask=next(stably_deact_mask_gen),
        unstable_mask=next(unstable_mask_gen),
        C=next(C_gen),
        transposed_layer=transposed_layer,
        b=b,
        transposed_layer_next=prev_layer.transposed_layer,
        P=next(P_gen),
        P_hat=next(P_hat_gen),
        p=next(p_gen),
    )
