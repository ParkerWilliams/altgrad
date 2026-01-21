"""W&B logging and alert callbacks.

Provides a tracker class for logging training metrics to Weights & Biases
with explicit step alignment and automatic alerts on stability threshold
breaches.

Key features:
  - Explicit step numbers in all logs (no auto-increment)
  - Configurable alerts for NaN, bit stall, overflow
  - Run resumption support via run_id

Example:
    >>> from altgrad.training.callbacks import WandbTracker
    >>> tracker = WandbTracker(config)
    >>> for step in range(config.max_steps):
    ...     metrics = train_step()
    ...     tracker.log_step(step, metrics)
    ...     action = tracker.check_alerts(step, metrics, config)
    ...     if action == 'stop':
    ...         break
    >>> tracker.finish()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from altgrad.training.config import TrainConfig

# Import wandb lazily to allow mocking in tests
_wandb = None


def _get_wandb():
    """Lazy import of wandb module."""
    global _wandb
    if _wandb is None:
        import wandb
        _wandb = wandb
    return _wandb


class WandbTracker:
    """Weights & Biases tracker with alert support.

    Handles W&B initialization, metric logging with explicit steps,
    and automatic alerts based on stability thresholds.

    Attributes:
        consecutive_nan_steps: Counter for consecutive NaN observations
        loss_history: Recent loss values for trend analysis
        _run: The wandb run object

    Example:
        >>> config = TrainConfig(project="my-project")
        >>> tracker = WandbTracker(config)
        >>> tracker.log_step(0, {"loss": 2.5})
        >>> tracker.finish()
    """

    def __init__(
        self,
        config: "TrainConfig",
        run_id: Optional[str] = None,
    ):
        """Initialize W&B tracker.

        Args:
            config: Training configuration with W&B settings
            run_id: Optional run ID for resuming a previous run
        """
        wandb = _get_wandb()

        # Convert config to dict for wandb
        if hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            config_dict = asdict(config)
        else:
            config_dict = dict(config) if hasattr(config, "__iter__") else {}

        # Initialize wandb
        init_kwargs = {
            "project": getattr(config, "project", "altgrad"),
            "config": config_dict,
        }

        if getattr(config, "run_name", None):
            init_kwargs["name"] = config.run_name

        if getattr(config, "tags", None):
            init_kwargs["tags"] = config.tags

        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"

        self._run = wandb.init(**init_kwargs)

        # Alert tracking
        self.consecutive_nan_steps = 0
        self.loss_history: list = []

    @property
    def run_id(self) -> str:
        """Get current run ID for checkpoint saving.

        Returns:
            W&B run ID string
        """
        return self._run.id

    def log_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log metrics with explicit step number.

        Args:
            step: Training step number
            metrics: Dictionary of metrics to log
        """
        wandb = _get_wandb()
        wandb.log(metrics, step=step)

    def check_alerts(
        self,
        step: int,
        metrics: Dict[str, Any],
        config: "TrainConfig",
    ) -> str:
        """Check metrics against stability thresholds and fire alerts.

        Checks for:
          - NaN in loss: increment counter, alert at nan_patience
          - High bit stall rate: warn if > bit_stall_threshold
          - High overflow rate: warn if > overflow_threshold

        Args:
            step: Current training step
            metrics: Current step metrics
            config: Training configuration with thresholds

        Returns:
            Action string:
              - 'continue': No issues
              - 'save_checkpoint': Anomaly detected, save checkpoint
              - 'stop': Critical issue, training should stop
        """
        wandb = _get_wandb()

        # Check for NaN loss
        loss = metrics.get("loss")
        if loss is not None:
            import math
            if math.isnan(loss):
                self.consecutive_nan_steps += 1

                nan_patience = getattr(config, "nan_patience", 10)
                if self.consecutive_nan_steps >= nan_patience:
                    wandb.alert(
                        title="Training stopped: NaN loss",
                        text=f"Loss was NaN for {self.consecutive_nan_steps} consecutive steps",
                        level=wandb.AlertLevel.ERROR,
                    )
                    return "stop"
                elif self.consecutive_nan_steps == 1:
                    return "save_checkpoint"
            else:
                self.consecutive_nan_steps = 0

        # Check bit stall rate
        bit_stall_rate = metrics.get("bit_stall_rate")
        bit_stall_threshold = getattr(config, "bit_stall_threshold", 0.5)
        if bit_stall_rate is not None and bit_stall_rate > bit_stall_threshold:
            wandb.alert(
                title="High bit stall rate",
                text=f"Bit stall rate {bit_stall_rate:.2%} exceeds threshold {bit_stall_threshold:.2%} at step {step}",
                level=wandb.AlertLevel.WARN,
                wait_duration=300,  # 5 minutes between alerts
            )

        # Check overflow rate
        overflow_rate = metrics.get("overflow_rate")
        overflow_threshold = getattr(config, "overflow_threshold", 0.01)
        if overflow_rate is not None and overflow_rate > overflow_threshold:
            wandb.alert(
                title="High overflow rate",
                text=f"Overflow rate {overflow_rate:.2%} exceeds threshold {overflow_threshold:.2%} at step {step}",
                level=wandb.AlertLevel.WARN,
                wait_duration=300,
            )

        return "continue"

    def finish(self) -> None:
        """Finish the W&B run."""
        wandb = _get_wandb()
        wandb.finish()


__all__ = [
    "WandbTracker",
]
