from abc import abstractmethod
from turtle import forward

import torch
from torch import Tensor, nn, tensor

from utils import bracket_minus, bracket_plus


class SolverOutputLayer(nn.Module):
    def __init__(self, H: Tensor, W_i: Tensor, initial_gamma: Tensor | None = None) -> None:
        super().__init__()
        self.H = H
        self.W_i = W_i

        if initial_gamma is not None:
            self.gamma: nn.Parameter = nn.Parameter(initial_gamma.clone().detach())
        else:
            self.gamma: nn.Parameter = nn.Parameter(torch.randn((H.size(0), 1)))
        assert self.gamma.shape == (H.size(0),)

    def forward(self) -> Tensor:
        H, gamma = self.H, self.gamma
        V_L = (-H.T @ gamma).squeeze()
        assert V_L.dim() == 1
        return V_L

    def clamp_parameters(self) -> None:
        self.gamma.clamp_(min=0)


class SolverIntermediateLayer(nn.Module):
    def __init__(
        self,
        W_i: Tensor,
        L_i: Tensor,
        U_i: Tensor,
        P_i: Tensor,
        P_hat_i: Tensor,
        p_i: Tensor,
        initial_pi_i: Tensor | None = None,
        initial_alpha_i: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.W_i = W_i
        self.L_i = L_i
        self.U_i = U_i
        self.P_i = P_i
        self.P_hat_i = P_hat_i
        self.p_i = p_i
        self.stably_act_mask: Tensor = L_i >= 0
        self.stably_deact_mask: Tensor = U_i <= 0
        self.unstable_mask: Tensor = (L_i < 0) & (U_i > 0)
        assert torch.all((self.stably_act_mask + self.stably_deact_mask + self.unstable_mask) == 1)

        self.num_neurons: int = self.W_i.size(0)
        self.num_unstable: int = int(self.unstable_mask.sum().item())
        self.C_i: Tensor = torch.zeros((self.num_neurons,))

        assert P_i.size(1) == P_hat_i.size(1) == self.num_unstable

        if initial_pi_i is not None:
            self.pi_i: nn.Parameter = nn.Parameter(initial_pi_i.clone().detach())
        else:
            self.pi_i: nn.Parameter = nn.Parameter(torch.randn((P_i.size(0),)))
        assert self.pi_i.shape == (P_i.size(0),)

        if initial_alpha_i is not None:
            self.alpha_i: nn.Parameter = nn.Parameter(initial_alpha_i.clone().detach())
        else:
            self.alpha_i: nn.Parameter = nn.Parameter(torch.randn((self.num_unstable,)))
        assert self.alpha_i.shape == (self.num_unstable,)

        self.V_hat_i: Tensor | None = None

    def set_target(self, j: int, is_min: bool) -> None:
        self.C_i: Tensor = torch.zeros((self.num_neurons,))
        self.C_i[j] = 1 if is_min else -1

    def get_V_hat(self) -> Tensor:
        assert self.V_hat_i is not None
        return self.V_hat_i

    def forward(self, V_next: Tensor, W_next: Tensor) -> Tensor:
        # fmt: off
        # Assign to local variables, so that they can be used w/o `self.` prefix.
        num_neurons, num_unstable, P_i, P_hat_i, C_i, stably_act_mask, stably_deact_mask, unstable_mask, pi_i, alpha_i, U_i, L_i = self.num_neurons, self.num_unstable, self.P_i, self.P_hat_i, self.C_i, self.stably_act_mask, self.stably_deact_mask, self.unstable_mask, self.pi_i, self.alpha_i, self.U_i, self.L_i
        # fmt: on

        V_i: Tensor = torch.zeros((num_neurons,))

        # Stably activated.
        stably_activated_V_i: Tensor = (V_next @ W_next.T).squeeze() - C_i
        V_i[stably_act_mask] = stably_activated_V_i[stably_act_mask]
        # torch.tensor(
        #     [V_next.T @ W_next[j] - C_i[j] for j in range(num_neurons)]
        # )

        # Stably deactivated.
        V_i[stably_deact_mask] = -C_i[stably_deact_mask]

        if num_unstable == 0:
            return V_i

        V_hat_i = (V_next @ W_next.T).squeeze()[unstable_mask] - pi_i @ P_hat_i
        self.V_hat_i = V_hat_i

        V_i[unstable_mask] = (
            (bracket_plus(V_hat_i) * U_i[unstable_mask]) / (U_i[unstable_mask] - L_i[unstable_mask])
            - C_i[unstable_mask]
            - alpha_i * bracket_minus(V_hat_i)
            - pi_i @ P_i
        )
        return V_i

    def get_obj_sum(self) -> Tensor:
        L_i, U_i, unstable_mask, p_i, pi_i = (
            self.L_i,
            self.U_i,
            self.unstable_mask,
            self.p_i,
            self.pi_i,
        )

        if self.num_unstable == 0:
            return torch.zeros((1,))

        V_hat_i = self.get_V_hat()
        return (
            torch.sum(
                (bracket_plus(V_hat_i) * U_i[unstable_mask] * L_i[unstable_mask])
                / (U_i[unstable_mask] - L_i[unstable_mask])
            )
            - pi_i @ p_i
        )

    def clamp_parameters(self) -> None:
        self.pi_i.clamp_(min=0)
        self.alpha_i.clamp_(min=0, max=1)
