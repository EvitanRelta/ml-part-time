from collections.abc import Iterator
from typing import Callable, Dict, Iterator, List, Tuple, TypeVar, Union

from torch import Tensor, fx, nn

from ..modules.solver_layers.input_layer import Input_SL
from ..modules.solver_layers.l1 import L1_SL
from ..modules.solver_layers.output_layer import Output_SL
from ..modules.solver_layers.relu import ReLU_SL
from ..preprocessing.graph_module_wrapper import GraphModuleWrapper
from ..preprocessing.named_solver_inputs import NamedSolverInputs
from . import preprocessing_utils
from .solver_inputs import SolverInputs
from .transpose import transpose_layer


def pick(tuple_index: int) -> Callable[[Tuple[Tensor, ...]], Tensor]:
    return lambda x: x[tuple_index]


pick_0 = pick(0)
pick_1 = pick(1)
pick_2 = pick(2)


def build_solver_graph_module(inputs: SolverInputs) -> fx.GraphModule:
    preprocessing_utils.freeze_model(inputs.model)

    graph_wrapper = GraphModuleWrapper(inputs.model, inputs.input_shape)
    unstable_masks = [(L < 0) & (U > 0) for L, U in zip(inputs.L_list, inputs.U_list)]

    # Initially set to solve for input layer.
    C_list, solve_coords = preprocessing_utils.get_C_for_layer(0, unstable_masks)

    named_solver_inputs = NamedSolverInputs(inputs, C_list)

    last_node = graph_wrapper.last_child
    solver_modules: Dict[str, nn.Module] = {}
    graph = fx.Graph()

    transposed_layer, bias_module = transpose_layer(
        last_node.module,
        last_node.input_shape,
        last_node.output_shape,
    )
    output_layer = Output_SL(
        transposed_layer=transposed_layer,
        bias_module=bias_module,
        L=named_solver_inputs.L_dict["output_layer"],
        U=named_solver_inputs.U_dict["output_layer"],
        C=named_solver_inputs.C_dict["output_layer"],
        H=inputs.H,
        d=inputs.d,
    )
    solver_modules["output_layer"] = output_layer
    prev_output = graph.call_module("output_layer")

    # Decompose the 3 outputs from current layer for the next layer.
    V_arg = graph.call_function(pick_0, (prev_output,))
    V_W_arg = graph.call_function(pick_1, (prev_output,))
    accum_sum_arg = graph.call_function(pick_2, (prev_output,))

    node = last_node
    while True:
        node = node.parent
        if node is None:
            break

        if isinstance(node.module, (nn.Linear, nn.Conv2d)):
            transposed_layer, bias_module = transpose_layer(
                node.module,
                node.input_shape,
                node.output_shape,
            )
            l1_solver_layer = L1_SL(
                transposed_layer=transposed_layer,
                bias_module=bias_module,
            )

            prev_output = graph.call_module(node.name, (V_arg, V_W_arg, accum_sum_arg))
            solver_modules[node.name] = l1_solver_layer

            # Decompose the 3 outputs from current layer for the next layer.
            V_arg = graph.call_function(pick_0, (prev_output,))
            V_W_arg = graph.call_function(pick_1, (prev_output,))
            accum_sum_arg = graph.call_function(pick_2, (prev_output,))
            continue

        if isinstance(node.module, nn.ReLU):
            relu_solver_layer = ReLU_SL(
                L=named_solver_inputs.L_dict[node.name],
                U=named_solver_inputs.U_dict[node.name],
                C=named_solver_inputs.C_dict[node.name],
                P=named_solver_inputs.P_dict[node.name],
                P_hat=named_solver_inputs.P_hat_dict[node.name],
                p=named_solver_inputs.p_dict[node.name],
            )

            prev_output = graph.call_module(node.name, (V_arg, V_W_arg, accum_sum_arg))
            solver_modules[node.name] = relu_solver_layer

            # Decompose the 3 outputs from current layer for the next layer.
            V_arg = graph.call_function(pick_0, (prev_output,))
            V_W_arg = graph.call_function(pick_1, (prev_output,))
            accum_sum_arg = graph.call_function(pick_2, (prev_output,))
            continue

        transposed_layer, _ = transpose_layer(
            node.module,
            node.input_shape,
            node.output_shape,
        )

        # Only feed this layer the `V_W` from previous layer.
        V_W_arg = graph.call_module(node.name, (V_W_arg,))
        solver_modules[node.name] = transposed_layer  # type: ignore
        continue

    solver_modules["input_layer"] = Input_SL(
        L=named_solver_inputs.L_dict["input_layer"],
        U=named_solver_inputs.U_dict["input_layer"],
        C=named_solver_inputs.C_dict["input_layer"],
    )
    prev_output = graph.call_module("input_layer", (V_arg, V_W_arg, accum_sum_arg))
    graph.output(prev_output)

    return fx.GraphModule(solver_modules, graph)


T = TypeVar("T")


def get_reversed_iterator(list_or_iterator: Union[List[T], Iterator[T]]) -> Iterator[T]:
    items = list(list_or_iterator)
    items.reverse()
    return iter(items)
