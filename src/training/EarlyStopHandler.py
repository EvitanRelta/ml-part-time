class EarlyStopHandler:
    """Handler to determine whether to early-stop the training.

    Based on Pytorch's `ReduceLROnPlateau` scheduler's patience /
    relative-threshold.
    """

    def __init__(self, patience: int, threshold: float) -> None:
        """
        Args:
            patience (int): Num. of epochs with no improvement, after which training should be stopped.
            threshold (float): Threshold to determine whether there's "no improvement". \
                No improvement is when `current_loss >= best_loss * (1 - threshold)`.
        """
        self.patience = patience
        self.threshold = threshold
        self._num_no_improvements: int = 0
        self._best_loss: float = float("inf")

    def is_early_stopped(self, current_loss: float) -> bool:
        """Returns whether to stop the training early."""
        has_no_improvement = current_loss >= self._best_loss * (1 - self.threshold)
        if has_no_improvement:
            self._num_no_improvements += 1
        else:
            self._best_loss = current_loss
            self._num_no_improvements = 0

        return self._num_no_improvements >= self.patience
