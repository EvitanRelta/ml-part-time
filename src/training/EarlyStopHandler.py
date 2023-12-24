class EarlyStopHandler:
    """Handler to determine whether to early-stop the training.

    Stops the training when the `ReduceLROnPlateau` scheduler's min LR has reached.
    """

    def __init__(self, min_lr: float) -> None:
        """
        Args:
            min_lr (float): Min LR of the `ReduceLROnPlateau` scheduler.
        """
        self.min_lr = min_lr

    def is_early_stopped(self, lr: float) -> bool:
        """Returns whether to stop the training early."""
        return lr == self.min_lr
