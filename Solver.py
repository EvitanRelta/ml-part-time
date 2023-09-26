from typing import cast

import torch
import torch.nn.functional as F
from torch import BoolTensor, Tensor, nn


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
    ):
        super().__init__()
        self.model = model
        self.L = L
        self.U = U
        self.H = H
        self.d = d
        self.P = P
        self.P_hat = P_hat
        self.p = p

        # Freeze model's layers.
        for param in model.parameters():
            param.requires_grad = False

        linear_layers = [
            layer for layer in model.children() if isinstance(layer, nn.Linear)
        ]
        self.num_layers: int = len(linear_layers)

        self.W: list[Tensor] = [torch.empty(0)] + [
            next(layer.parameters()) for layer in linear_layers
        ]
        self.b: list[Tensor] = [torch.empty(0)] + [
            list(layer.parameters())[1] for layer in linear_layers
        ]

        template: list[Tensor] = (
            [torch.empty(0)]
            + [torch.randn((layer.out_features,)) for layer in linear_layers[:-1]]
            + [torch.empty(0)]
        )

        self.C: list[Tensor] = [torch.zeros((linear_layers[0].in_features,))] + [
            torch.zeros((layer.out_features,)) for layer in linear_layers
        ]

        # Assuming minimising of x4
        self.C[1][1] = 1

        self.stably_act_masks: list[Tensor] = [x >= 0 for x in L]
        self.stably_deact_masks: list[Tensor] = [x <= 0 for x in U]
        self.unstable_masks: list[Tensor] = [
            (L[i] < 0) & (U[i] > 0) for i in range(len(U))
        ]

        self.gamma = nn.Parameter(torch.randn(7, 1))
        self.pi = cast(
            list[nn.Parameter],
            nn.ParameterList(
                [torch.empty(0)]
                + [
                    torch.randn((P[i + 1].shape[0],))
                    for i in range(len(linear_layers) - 1)
                ]
                + [torch.empty(0)]
            ),
        )
        self.alpha = cast(
            list[nn.Parameter],
            nn.ParameterList([torch.randn_like(x) for x in template]),
        )

    def forward(self) -> Tensor:
        cls = self.__class__
        # fmt: off

        # Assign to local variables, so that they can be used w/o `self.` prefix.
        (L, U, H, d, P, P_hat, p, num_layers, W, b, C, stably_act_masks, stably_deact_masks, unstable_masks, gamma, pi, alpha) = (self.L, self.U, self.H, self.d, self.P, self.P_hat, self.p, self.num_layers, self.W, self.b, self.C, self.stably_act_masks, self.stably_deact_masks, self.unstable_masks, self.gamma, self.pi, self.alpha)
        # fmt: on
        l = num_layers
        V: list[Tensor] = [None] * (l + 1)  # type: ignore
        V[l] = (-H.T @ gamma).squeeze()
        assert (
            V[l].squeeze().shape == C[l].shape
        ), f"V[l].shape == {V[l].shape}, C[l].shape == {C[l].shape}"

        bounds_frac: list[Tensor] = [None] * (l + 1)  # type: ignore
        for i in range(l - 1, 0, -1):  # From l-1 to 1 (inclusive)
            stably_activated_V_i: Tensor = torch.tensor(
                [V[i + 1].T @ W[i + 1][j] - C[i][j] for j in range(len(C[i]))]
            )
            # A potentially more efficient equation is:
            # stably_activated_V_i = (V[i + 1].T @ W[i + 1].T).squeeze() - C[i]
            assert stably_activated_V_i.dim() == 1
            print("stably_activated_V_i:\n", stably_activated_V_i)

            stably_deactivated_V_i: Tensor = -C[i]
            assert stably_activated_V_i.dim() == 1
            print("stably_deactivated_V_i:\n", stably_deactivated_V_i)

            # unstable_V_hat_i: Tensor = V[i + 1].T @ W[i + 1] - pi[i].T @ P_hat[i]
            unstable_V_hat_i: Tensor = torch.tensor(
                [
                    V[i + 1].T @ W[i + 1][j] - pi[i].T @ P_hat[i][:, j]
                    for j in range(len(C[i]))
                ]
            )
            assert unstable_V_hat_i.dim() == 1

            frac = (cls.bracket_plus(unstable_V_hat_i) + U[i]) / (U[i] - L[i])
            bounds_frac[i] = frac
            unstable_V_i: Tensor = (
                frac
                - C[i]
                - alpha[i] @ cls.bracket_minus(unstable_V_hat_i)
                - pi[i].T @ P[i]
            )
            assert unstable_V_i.dim() == 1

            V[i] = (
                (stably_act_masks[i] * stably_activated_V_i)
                + (stably_deact_masks[i] * stably_deactivated_V_i)
                + (unstable_masks[i] * unstable_V_i)
            )
            assert V[i].dim() == 1

        temp_2 = C[0].T - V[1].T @ W[1]
        print([(V[i].T.shape, b[i].shape) for i in range(1, l + 1)])
        max_objective: Tensor = (
            (F.relu(temp_2) @ L[0])
            - (F.relu(-temp_2) @ U[0])
            + gamma.T @ d
            - torch.stack([V[i].T @ b[i] for i in range(1, l + 1)]).sum(dim=0)
            + self.last_dbl_summation(bounds_frac, l)
        )
        loss = -max_objective

        return loss.sum()

    def last_dbl_summation(
        self,
        bounds_frac: list[Tensor],
        l: int,
    ) -> Tensor:
        output: Tensor = torch.zeros((1,))
        for i in range(1, l):
            frac = bounds_frac[i]
            output += torch.sum(
                self.unstable_masks[i] * (frac - self.pi[i] @ self.p[i])
            )
        return output

    @staticmethod
    def bracket_plus(X: Tensor) -> Tensor:
        return torch.clamp(X, min=0)

    @staticmethod
    def bracket_minus(X: Tensor) -> Tensor:
        return -torch.clamp(X, max=0)
