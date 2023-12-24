from dataclasses import dataclass
from typing import List, Optional, Tuple

from torch import fx, nn

from .linearize_conv2d import compute_conv2d_output_shape


@dataclass
class GraphModuleWrapper:
    graph_module: fx.GraphModule
    input_shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        input_node: fx.Node = next(iter(self.graph_module.graph.nodes))
        assert len(input_node.users) == 1, "Failed the assumption that there's only 1 input layer."

        first_layer_node = next(iter(input_node.users))
        self.first_child = NodeWrapper(first_layer_node, self.input_shape, self, parent=None)

        self.last_child = self.first_child
        while len(self.last_child.children) > 0:
            self.last_child = self.last_child.children[0]

    def get_module(self, node: fx.Node) -> nn.Module:
        """Get the underlying PyTorch module from a node."""
        module_name = node.target
        assert isinstance(module_name, str)
        return self.graph_module.get_submodule(module_name)

    def __repr__(self) -> str:
        return self.graph_module.__repr__()


@dataclass
class NodeWrapper:
    node: fx.Node
    input_shape: Tuple[int, ...]
    graph_module_wrapper: GraphModuleWrapper
    parent: Optional["NodeWrapper"]

    def __post_init__(self) -> None:
        assert is_module(self.node), f"`node={self.node}` doesn't represent a PyTorch module."
        self.output_shape: Tuple[int, ...] = compute_output_shape(self.module, self.input_shape)
        self.children: List["NodeWrapper"] = [
            NodeWrapper(node, self.output_shape, self.graph_module_wrapper, parent=self)
            for node in self.node.users
            if is_module(node)
        ]

    @property
    def name(self) -> str:
        return str(self.node)

    @property
    def module(self) -> nn.Module:
        return self.graph_module_wrapper.get_module(self.node)

    def __repr__(self) -> str:
        self_repr = self.node.__repr__()
        parent_repr = self.parent.node.__repr__() if self.parent is not None else "None"
        children_repr = f"[{', '.join(x.node.__repr__() for x in self.children)}]"
        return f"NodeWrapper(node={self_repr}, input_shape={self.input_shape}, output_shape={self.output_shape}, parent={parent_repr}, children={children_repr})"


def is_module(node: fx.Node) -> bool:
    """Whether a `torch.fx.Node` represents a PyTorch module."""
    return node.op == "call_module"


def compute_output_shape(module: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if isinstance(module, nn.Linear):
        assert len(input_shape) <= 2
        is_batched = len(input_shape) == 2
        return (input_shape[0], module.out_features) if is_batched else (module.out_features,)

    if isinstance(module, nn.Conv2d):
        assert len(input_shape) == 4 or len(input_shape) == 3
        is_batched = len(input_shape) == 4
        return (
            (input_shape[0], *compute_conv2d_output_shape(module, input_shape[1:]))
            if is_batched
            else compute_conv2d_output_shape(module, input_shape)
        )
    if isinstance(module, nn.Flatten):
        start_dim = (
            module.start_dim if module.start_dim >= 0 else len(input_shape) + module.start_dim
        )
        end_dim = module.end_dim if module.end_dim >= 0 else len(input_shape) + module.end_dim

        # Ensure the dims are within the bounds of input_shape
        assert 0 <= start_dim < len(input_shape), "start_dim out of range"
        assert (
            start_dim <= end_dim < len(input_shape)
        ), "end_dim out of range or less than start_dim"

        # Calculate the number of elements in the flattened part
        flattened_elements = 1
        for dim in input_shape[start_dim : end_dim + 1]:
            flattened_elements *= dim

        return input_shape[:start_dim] + (flattened_elements,) + input_shape[end_dim + 1 :]

    if isinstance(module, (nn.ReLU, nn.BatchNorm2d)):
        return input_shape
    raise NotImplementedError()
