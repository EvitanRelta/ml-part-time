import torch
import torch.nn.functional as F
from torch import Tensor, nn

from SolverLayer import SolverIntermediateLayer, SolverOutputLayer


class Solver(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        L: list[Tensor],
        U: list[Tensor],
        H: Tensor,
        d: Tensor,
        P: list[Tensor],
        P_hat: list[Tensor],
        p: list[Tensor],
        initial_gamma: Tensor | None = None,
        initial_pi: list[Tensor] | None = None,
        initial_alpha: list[Tensor] | None = None,
    ):
        super().__init__()
        cls = self.__class__

        self.model = model
        self.L = L
        self.U = U
        self.H = H
        self.d = d
        self.P = P
        self.P_hat = P_hat
        self.p = p

        self.num_layers, self.W, self.b = cls.decompose_model(model)
        self.C_0 = torch.zeros((self.W[1].size(1),))

        self.layers = nn.ModuleList([nn.Identity()])  # Set index-0 to be some placeholder Module.

        for i in range(1, self.num_layers):
            initial_pi_i = initial_pi[i - 1] if initial_pi is not None else None
            initial_alpha_i = initial_alpha[i - 1] if initial_alpha is not None else None
            self.layers.append(
                SolverIntermediateLayer(
                    self.W[i],
                    self.W[i + 1],
                    L[i],
                    U[i],
                    P[i - 1],
                    P_hat[i - 1],
                    p[i - 1],
                    initial_pi_i,
                    initial_alpha_i,
                )
            )

        self.layers.append(SolverOutputLayer(self.W[-1], H, initial_gamma))

        layer = self.layers[1]
        assert isinstance(layer, SolverIntermediateLayer)
        layer.set_target(1, is_min=True)

    @staticmethod
    def decompose_model(
        model: nn.Module,
    ) -> tuple[int, list[Tensor], list[Tensor]]:
        # Freeze model's layers.
        for param in model.parameters():
            param.requires_grad = False

        linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
        num_layers: int = len(linear_layers)

        W: list[Tensor] = [torch.empty(0)] + [next(layer.parameters()) for layer in linear_layers]
        b: list[Tensor] = [torch.empty(0)] + [
            list(layer.parameters())[1] for layer in linear_layers
        ]
        return num_layers, W, b

    def forward(self) -> Tensor:
        L, U, d, W, b = self.L, self.U, self.d, self.W, self.b
        l = self.num_layers
        V: list[Tensor] = [None] * (l + 1)  # type: ignore
        self.V = V
        last_layer = self.layers[l]
        assert isinstance(last_layer, SolverOutputLayer)
        V[l] = last_layer.forward()

        for i in range(l - 1, 0, -1):  # From l-1 to 1 (inclusive)
            layer = self.layers[i]
            assert isinstance(layer, SolverIntermediateLayer)
            V[i] = layer.forward(V[i + 1])

        temp_2 = self.C_0.T - V[1].T @ W[1]
        max_objective: Tensor = (
            (F.relu(temp_2) @ L[0])
            - (F.relu(-temp_2) @ U[0])
            + last_layer.gamma.T @ d
            - torch.stack([V[i].T @ b[i] for i in range(1, l + 1)]).sum(dim=0)
            + self.last_dbl_summation(l)
        )
        loss = -max_objective

        return loss.sum()

    def last_dbl_summation(
        self,
        l: int,
    ) -> Tensor:
        output: Tensor = torch.zeros((1,))
        for i in range(1, l):
            layer = self.layers[i]
            assert isinstance(layer, SolverIntermediateLayer)
            output += layer.get_obj_sum()
        return output

    def clamp_parameters(self):
        with torch.no_grad():
            # Ensure all elements of solver.gamma are >= 0
            last_layer = self.layers[-1]
            assert isinstance(last_layer, SolverOutputLayer)
            last_layer.clamp_parameters()

            for i in range(1, self.num_layers):
                layer = self.layers[i]
                assert isinstance(layer, SolverIntermediateLayer)
                layer.clamp_parameters()
