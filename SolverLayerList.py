from typing import Iterator, Literal, Sequence, overload

from torch import nn

from inputs_dataclasses import SolverInputs
from SolverLayer import (
    SolverInputLayer,
    SolverIntermediateLayer,
    SolverLayer,
    SolverOutputLayer,
)


class SolverLayerList(nn.Module, Sequence):
    def __len__(self) -> int:
        return self.num_layers + 1

    def __iter__(self) -> Iterator[SolverLayer]:
        return super().__iter__()

    @overload
    def __getitem__(self, i: Literal[-1]) -> SolverOutputLayer:
        ...

    @overload
    def __getitem__(self, i: Literal[0]) -> SolverInputLayer:
        ...

    @overload
    def __getitem__(self, i: int) -> SolverIntermediateLayer:
        ...

    def __getitem__(self, i: int) -> SolverLayer:
        if i < -1 or i >= len(self):
            raise IndexError(f"Expected an index i, where -1 <= i < {len(self)}, but got {i}.")

        layer: nn.Module = self._layers[i]
        assert isinstance(layer, SolverLayer)
        return layer

    def __init__(self, inputs: SolverInputs):
        super().__init__()
        self.num_layers = inputs.num_layers

        self._layers = nn.ModuleList([SolverInputLayer(inputs[0])])
        for i in range(1, self.num_layers):
            self._layers.append(SolverIntermediateLayer(inputs[i]))
        self._layers.append(SolverOutputLayer(inputs[-1]))
