from collections.abc import Iterator
from typing import Iterator, List, Tuple, TypeVar, Union

import pytest
from torch import Tensor, nn

from ..modules.solver_layers.base_class import SolverLayer
from ..modules.solver_layers.input_layer import InputLayer
from ..modules.solver_layers.intermediate_layer import IntermediateLayer
from ..modules.solver_layers.output_layer import OutputLayer
from . import preprocessing_utils
from .solver_inputs import SolverInputs
from .transpose import transpose_layer


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
    transposed_layer, bias_module, out_feat = transpose_layer(last_layer, last_layer.out_features)

    output_layer = OutputLayer(
        L=next(L_gen),
        U=next(U_gen),
        stably_act_mask=next(stably_act_mask_gen),
        stably_deact_mask=next(stably_deact_mask_gen),
        unstable_mask=next(unstable_mask_gen),
        C=next(C_gen),
        transposed_layer=transposed_layer,
        bias_module=bias_module,
        H=inputs.H,
        d=inputs.d,
    )
    solver_layers: List[SolverLayer] = [output_layer]

    prev_layer: Union[OutputLayer, IntermediateLayer] = output_layer
    prev_out_feat: int = out_feat

    while True:
        try:
            prev_layer, prev_out_feat = build_intermediate_layer(
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
                prev_out_feat=prev_out_feat,
            )
            solver_layers.append(prev_layer)
        except StopIteration:
            break

    solver_layers.append(
        InputLayer(
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


T = TypeVar("T")


def get_reversed_iterator(list_or_iterator: Union[List[T], Iterator[T]]) -> Iterator[T]:
    items = list(list_or_iterator)
    items.reverse()
    return iter(items)


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
    prev_layer: Union[IntermediateLayer, OutputLayer],
    prev_out_feat: int,
) -> Tuple[IntermediateLayer, int]:
    layer = next(layer_gen)
    while not isinstance(layer, (nn.Linear, nn.Conv2d)):
        layer = next(layer_gen)

    transposed_layer, bias_module, out_feat = transpose_layer(layer, prev_out_feat)
    return (
        IntermediateLayer(
            L=next(L_gen),
            U=next(U_gen),
            stably_act_mask=next(stably_act_mask_gen),
            stably_deact_mask=next(stably_deact_mask_gen),
            unstable_mask=next(unstable_mask_gen),
            C=next(C_gen),
            transposed_layer=transposed_layer,
            bias_module=bias_module,
            transposed_layer_next=prev_layer.transposed_layer,
            P=next(P_gen),
            P_hat=next(P_hat_gen),
            p=next(p_gen),
        ),
        out_feat,
    )
