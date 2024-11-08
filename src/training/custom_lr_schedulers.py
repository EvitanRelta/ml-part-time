import math
from typing import List, Optional, Union

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer


class CosineAnnealingWarmRestartsWithDecay(CosineAnnealingWarmRestarts):
    """Cosine annealing scheduler with warm restarts and amplitude decay.

    Extends PyTorch's CosineAnnealingWarmRestarts by adding an exponential decay
    factor to the learning rate after each restart.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        decay_factor: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        """
        Rest of the args are the same as `CosineAnnealingWarmRestarts`.
        Args:
            decay_factor (float, optional): Factor by which to decay the maximum learning
                rate after each restart. Must be > 0 and <= 1. Default: 0.5.
        """
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.decay_factor = decay_factor

    def get_lr(self) -> List[float]:
        if self.T_mult == 1:
            restart_count = self.last_epoch // self.T_0
        else:
            restart_count = math.floor(math.log(1 + self.last_epoch / self.T_0, self.T_mult))

        decay = self.decay_factor**restart_count
        lr_list = super().get_lr()
        return [lr * decay for lr in lr_list]  # type: ignore


class ReduceLROnRecentPlateau(ReduceLROnPlateau):
    """Reduce learning rate when a metric has stopped improving over recent epochs.

    Extends PyTorch's ReduceLROnPlateau by considering only the most recent
    `2 * patience` window of loss values when determining if the model has plateaued.

    Only the mode `mode="min"`, `threshold_mode="rel"` is implemented.
    """

    def __init__(
        self,
        optimizer,
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        """
        Args are the same as `ReduceLROnPlateau`.
        """
        super().__init__(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode="rel",
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )
        self.loss_window: List[float] = []
        self.window_size = 2 * patience
        self.num_since_better = 0

    def is_better(self, a, best):
        if super().is_better(a, best):
            self.num_since_better = 0
            return True
        self.num_since_better += 1
        return False

    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        if self.num_since_better >= self.window_size:
            self.best = self.loss_window[-self.window_size]
        super().step(metrics, epoch)
        self.loss_window.append(float(metrics))
