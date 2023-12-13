from dataclasses import dataclass

from dataclass_wizard import YAMLWizard
from typing_extensions import override


@dataclass
class TrainingConfig(YAMLWizard):
    # ==========================================================================
    #                      Optimizer & LR-scheduler configs
    # ==========================================================================
    max_lr: float = 1
    """Max learning-rate. The starting LR given to the `Adam` optimizer. Defaults to 1."""
    min_lr: float = 1e-6
    """Min learning-rate to decay until.
    The `min_lr` param used by the `ReduceLROnPlateau` scheduler. Defaults to 1e-6."""
    reduce_lr_factor: float = 0.5
    """Factor by which the learning rate will be reduced.
    The `factor` param used by the `ReduceLROnPlateau` scheduler. Defaults to 0.5"""
    reduce_lr_patience: int = 3
    """Number of epochs with no improvement after which learning rate will be reduced.
    The `patience` param used by the `ReduceLROnPlateau` scheduler. Defaults to 3."""
    reduce_lr_threshold: float = 1e-3
    """Threshold for measuring the new optimum, to only focus on significant changes.
    The `threshold` param used by the `ReduceLROnPlateau` scheduler. Defaults to 1e-3."""

    # ==========================================================================
    #                           Early-stopping configs
    # ==========================================================================
    stop_patience: int = 40
    """Num. of epochs with no improvement, after which training should be stopped.
    Defaults to 40."""
    stop_threshold: float = 1e-3
    """Threshold to determine whether there's "no improvement" for early-stopping.
    No improvement is when `current_loss >= best_loss * (1 - threshold)`.
    Defaults to 1e-3."""

    # ==========================================================================
    #                                Misc. configs
    # ==========================================================================
    disable_adv_check: bool = False
    """Whether to disable the adversarial check. Defaults to False."""
    num_epoch_adv_check: int = 10
    """Perform adversarial check every `num_epoch_adv_check` epochs. Only has an effect when
    `disable_adv_check=False`. Defaults to 10."""
    disable_progress_bar: bool = False
    """Whether to disable tqdm's progress bar during training. Defaults to False."""

    @override
    @classmethod
    def from_yaml_file(cls, file: str) -> "TrainingConfig":
        """Reads the YAML file's contents and converts to an instance of `TrainingConfig`."""
        return super().from_yaml_file(file)  # type: ignore
