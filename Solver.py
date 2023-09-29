from inspect import Parameter
from typing import cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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

        self.linear_layers, self.num_layers, self.W, self.b = cls.decompose_model(model)

        self.C: list[Tensor] = [torch.zeros((self.linear_layers[0].in_features,))] + [
            torch.zeros((layer.out_features,)) for layer in self.linear_layers
        ]

        # Assuming minimising of x4
        self.C[1][1] = 1

        (
            self.stably_act_masks,
            self.stably_deact_masks,
            self.unstable_masks,
        ) = cls.get_stability_masks(L, U)

        self.init_parameters(
            self.H,
            self.linear_layers,
            self.P,
            self.unstable_masks,
            initial_gamma,
            initial_pi,
            initial_alpha,
        )

    @staticmethod
    def decompose_model(
        model: nn.Module,
    ) -> tuple[list[nn.Linear], int, list[Tensor], list[Tensor]]:
        # Freeze model's layers.
        for param in model.parameters():
            param.requires_grad = False

        linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
        num_layers: int = len(linear_layers)

        W: list[Tensor] = [torch.empty(0)] + [next(layer.parameters()) for layer in linear_layers]
        b: list[Tensor] = [torch.empty(0)] + [
            list(layer.parameters())[1] for layer in linear_layers
        ]
        return linear_layers, num_layers, W, b

    @staticmethod
    def get_stability_masks(
        L: list[Tensor], U: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        stably_act_masks: list[Tensor] = [x >= 0 for x in L]
        stably_deact_masks: list[Tensor] = [x <= 0 for x in U]
        unstable_masks: list[Tensor] = [(L[i] < 0) & (U[i] > 0) for i in range(len(U))]
        return stably_act_masks, stably_deact_masks, unstable_masks

    def init_parameters(
        self,
        H: Tensor,
        linear_layers: list[nn.Linear],
        P: list[Tensor],
        unstable_masks: list[Tensor],
        initial_gamma: Tensor | None = None,
        initial_pi: list[Tensor] | None = None,
        initial_alpha: list[Tensor] | None = None,
    ) -> None:
        num_intermediate_layers: int = len(linear_layers) - 1

        if initial_gamma is not None:
            self.gamma: nn.Parameter = nn.Parameter(initial_gamma)
        else:
            self.gamma: nn.Parameter = nn.Parameter(torch.randn((H.size(0), 1)))
        assert self.gamma.shape == (H.size(0),)

        if initial_pi is not None:
            self.pi: list[nn.Parameter] = cast(list[nn.Parameter], initial_pi)
        else:
            _pi = nn.ParameterList([torch.randn((X.size(0),)) for X in P])
            # Casting `ParameterList` into `list[Parameter]` for easier typing later.
            # (becuz `ParameterList.__getitem__` returns type `Any``)
            self.pi: list[nn.Parameter] = cast(list[nn.Parameter], _pi)
        assert len(self.pi) == num_intermediate_layers
        assert len(self.pi) == len(P)
        for i in range(len(self.pi)):
            assert self.pi[i].shape == (P[i].size(0),)

        if initial_alpha is not None:
            self.alpha: list[nn.Parameter] = cast(list[nn.Parameter], initial_alpha)
        else:
            _alpha = nn.ParameterList(
                [
                    torch.randn((cast(int, unstable_masks[i + 1].sum().item()),))
                    for i in range(num_intermediate_layers)
                ]
            )
            # Casting `ParameterList` into `list[Parameter]` for easier typing later.
            # (becuz `ParameterList.__getitem__` returns type `Any``)
            self.alpha: list[nn.Parameter] = cast(list[nn.Parameter], _alpha)
        assert len(self.alpha) == num_intermediate_layers
        for i in range(num_intermediate_layers):
            num_unstable = unstable_masks[i + 1].sum().item()  # exclude input/output layers
            assert isinstance(num_unstable, int)
            assert self.alpha[i].shape == (num_unstable,)

    def forward(self) -> Tensor:
        cls = self.__class__
        # fmt: off
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        (L, U, H, d, P, P_hat, p, num_layers, W, b, C, stably_act_masks, stably_deact_masks, unstable_masks, gamma, pi, alpha) = (self.L, self.U, self.H, self.d, self.P, self.P_hat, self.p, self.num_layers, self.W, self.b, self.C, self.stably_act_masks, self.stably_deact_masks, self.unstable_masks, self.gamma, self.pi, self.alpha)
        # fmt: on
        l = num_layers
        V: list[Tensor] = [None] * (l + 1)  # type: ignore
        V[l] = (-H.T @ gamma).squeeze()
        assert V[l].shape == C[l].shape

        bounds_frac: list[Tensor] = [None] * (l + 1)  # type: ignore
        for i in range(l - 1, 0, -1):  # From l-1 to 1 (inclusive)
            stably_activated_V_i: Tensor = torch.tensor(
                [V[i + 1].T @ W[i + 1][j] - C[i][j] for j in range(len(C[i]))]
            )
            # A potentially more efficient equation is:
            # stably_activated_V_i = (V[i + 1].T @ W[i + 1].T).squeeze() - C[i]
            assert stably_activated_V_i.dim() == 1
            # print(f"stably_activated_V[{i}]:\n", stably_activated_V_i, "\n")

            stably_deactivated_V_i: Tensor = -C[i]
            assert stably_activated_V_i.dim() == 1
            # print(f"stably_deactivated_V[{i}]:\n", stably_deactivated_V_i, "\n")

            # unstable_V_hat_i: Tensor = V[i + 1].T @ W[i + 1] - pi[i].T @ P_hat[i - 1]
            unstable_V_hat_i: Tensor = torch.tensor(
                [
                    V[i + 1].T @ W[i + 1][j] - pi[i - 1].T @ P_hat[i - 1][:, j]
                    for j in range(len(C[i]))
                ]
            )
            assert unstable_V_hat_i.dim() == 1
            # print(f"unstable_V_hat[{i}]:\n", unstable_V_hat_i, "\n")

            frac = (cls.bracket_plus(unstable_V_hat_i) + U[i]) / (U[i] - L[i])
            bounds_frac[i] = frac
            _full_alpha: Tensor = torch.zeros_like(C[i])
            _full_alpha[unstable_masks[i]] = alpha[i - 1]
            unstable_V_i: Tensor = (
                frac
                - C[i]
                - _full_alpha @ cls.bracket_minus(unstable_V_hat_i)
                - pi[i - 1].T @ P[i - 1]
            )
            assert unstable_V_i.dim() == 1
            # print(f"unstable_V[{i}]:\n", unstable_V_i, "\n")

            V[i] = (
                (stably_act_masks[i] * stably_activated_V_i)
                + (stably_deact_masks[i] * stably_deactivated_V_i)
                + (unstable_masks[i] * unstable_V_i)
            )
            assert V[i].dim() == 1
            # print(f"V[{i}]:\n", V[i], "\n")

        # print(f"V:\n", V, "\n")

        temp_2 = C[0].T - V[1].T @ W[1]
        max_objective: Tensor = (
            (F.relu(temp_2) @ L[0])
            - (F.relu(-temp_2) @ U[0])
            + gamma.T @ d
            - torch.stack([V[i].T @ b[i] for i in range(1, l + 1)]).sum(dim=0)
            + self.last_dbl_summation(bounds_frac, l, unstable_masks, pi, p)
        )
        loss = -max_objective

        return loss.sum()

    @classmethod
    def last_dbl_summation(
        cls,
        bounds_frac: list[Tensor],
        l: int,
        unstable_masks: list[Tensor],
        pi: list[nn.Parameter],
        p: list[Tensor],
    ) -> Tensor:
        output: Tensor = torch.zeros((1,))
        for i in range(1, l):
            frac = bounds_frac[i]
            output += torch.sum(unstable_masks[i] * (frac - pi[i - 1] @ p[i - 1]))
        return output

    def clamp_parameters(self):
        with torch.no_grad():
            # Ensure all elements of solver.gamma are >= 0
            self.gamma.clamp_(min=0)

            # Ensure all elements of solver.pi are >= 0
            for pi in self.pi:
                pi.clamp_(min=0)

            # Ensure all elements of solver.alpha are between 0 and 1
            for alpha in self.alpha:
                alpha.clamp_(min=0, max=1)

    @staticmethod
    def bracket_plus(X: Tensor) -> Tensor:
        return torch.clamp(X, min=0)

    @staticmethod
    def bracket_minus(X: Tensor) -> Tensor:
        return -torch.clamp(X, max=0)
