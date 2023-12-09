from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # ==========================================================================
    #                      Optimizer & LR-scheduler configs
    # ==========================================================================
    max_lr: float = 2
    """Max learning-rate. The starting LR given to the `Adam` optimizer. Defaults to 2."""
    min_lr: float = 1e-6
    """Min learning-rate to decay until.
    The `min_lr` param used by the `ReduceLROnPlateau` scheduler. Defaults to 1e-6."""
    reduce_lr_factor: float = 0.5
    """Factor by which the learning rate will be reduced.
    The `factor` param used by the `ReduceLROnPlateau` scheduler. Defaults to 0.5"""
    reduce_lr_patience: int = 3
    """Number of epochs with no improvement after which learning rate will be reduced.
    The `patience` param used by the `ReduceLROnPlateau` scheduler. Defaults to 3."""
    reduce_lr_threshold: float = 0.001
    """Threshold for measuring the new optimum, to only focus on significant changes.
    The `threshold` param used by the `ReduceLROnPlateau` scheduler. Defaults to 0.001."""

    # ==========================================================================
    #                           Early-stopping configs
    # ==========================================================================
    stop_patience: int = 10
    """Num. of epochs with no improvement, after which training should be stopped.
    Defaults to 10."""
    stop_threshold: float = 1e-3
    """Threshold to determine whether there's "no improvement" for early-stopping.
    No improvement is when `current_loss >= best_loss * (1 - threshold)`.
    Defaults to 1e-3."""

    # ==========================================================================
    #                                Misc. configs
    # ==========================================================================
    run_adv_check: bool = True
    """Whether to run the adversarial check. Defaults to True."""
    num_epoch_adv_check: int = 10
    """Perform adversarial check every `num_epoch_adv_check` epochs. Only has an effect when
    `run_adv_check=True`. Defaults to 10."""
    disable_progress_bar: bool = False
    """Whether to disable tqdm's progress bar during training. Defaults to False."""
