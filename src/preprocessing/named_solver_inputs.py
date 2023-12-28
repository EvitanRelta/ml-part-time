from typing import Dict, List, Tuple

from torch import Tensor, nn

from .graph_module_wrapper import GraphModuleWrapper
from .solver_inputs import SolverInputs


class NamedSolverInputs:
    def __init__(self, inputs: SolverInputs, C_list: List[Tensor]) -> None:
        self.H = inputs.H
        self.d = inputs.d

        graph_wrapper = GraphModuleWrapper(inputs.model, inputs.input_shape)
        self.C_dict: Dict[str, Tensor] = {
            "input_layer": inputs.L_list[0],
        }
        self.L_dict: Dict[str, Tensor] = {
            "input_layer": inputs.L_list[0],
        }
        self.U_dict: Dict[str, Tensor] = {
            "input_layer": inputs.U_list[0],
        }
        self.P_dict: Dict[str, Tensor] = {}
        self.P_hat_dict: Dict[str, Tensor] = {}
        self.p_dict: Dict[str, Tensor] = {}
        self.input_shapes: Dict[str, Tuple[int, ...]] = {}
        self.output_shapes: Dict[str, Tuple[int, ...]] = {}

        relu_nodes = (x for x in graph_wrapper if isinstance(x.module, nn.ReLU))
        for i, relu_node in enumerate(relu_nodes, start=1):
            relu_name = relu_node.name
            self.C_dict[relu_name] = C_list[i]
            self.L_dict[relu_name] = inputs.L_list[i]
            self.U_dict[relu_name] = inputs.U_list[i]
            self.P_dict[relu_name] = inputs.P_list[i - 1]
            self.P_hat_dict[relu_name] = inputs.P_hat_list[i - 1]
            self.p_dict[relu_name] = inputs.p_list[i - 1]

        for node in graph_wrapper:
            node_name = node.name
            self.input_shapes[node_name] = node.input_shape
            self.output_shapes[node_name] = node.output_shape
