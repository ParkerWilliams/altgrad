"""Training configuration dataclass and YAML serialization.

Provides a comprehensive configuration for training runs including model
architecture, training hyperparameters, quantization settings, checkpointing,
stability thresholds, logging intervals, and W&B integration.

Example:
    >>> from altgrad.training.config import TrainConfig, load_config, save_config
    >>> config = TrainConfig(n_layer=6, n_head=6, n_embd=384)
    >>> save_config(config, 'config.yaml')
    >>> loaded = load_config('config.yaml')
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class TrainConfig:
    """Complete training run configuration.

    Groups related settings:
      - Model architecture (n_layer, n_head, etc.)
      - Training hyperparameters (batch_size, learning_rate, etc.)
      - Quantization settings (use_fp8, fp8_format, use_shadow)
      - Checkpointing (interval, directory, max checkpoints)
      - Stability thresholds (nan_patience, bit_stall, overflow, dead neuron)
      - Logging intervals (log_interval, eval_interval)
      - W&B configuration (project, run_name, tags)
      - Reproducibility (seed)

    Attributes:
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        block_size: Context length (sequence length)
        vocab_size: Vocabulary size
        dropout: Dropout probability

        batch_size: Training batch size
        learning_rate: Initial learning rate
        max_steps: Maximum training steps
        warmup_steps: Learning rate warmup steps
        grad_clip: Gradient clipping threshold (0 to disable)

        use_fp8: Enable FP8 quantization
        fp8_format: FP8 format name (E5M2, E3M4, etc.)
        use_shadow: Keep FP32 shadow copy for comparison

        checkpoint_interval: Steps between checkpoints
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum checkpoints to retain

        nan_patience: Steps with NaN loss before stopping
        bit_stall_threshold: Alert if stall rate exceeds this
        overflow_threshold: Alert if overflow rate exceeds this
        dead_neuron_threshold: Gradient magnitude below which neuron is "dead"
        dead_neuron_window: Steps over which to track dead neurons

        log_interval: Steps between metric logging
        eval_interval: Steps between validation runs

        project: W&B project name
        run_name: W&B run name
        tags: W&B tags for run categorization

        seed: Random seed for reproducibility
    """

    # Model architecture
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    vocab_size: int = 50304
    dropout: float = 0.0

    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_steps: int = 5000
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Quantization settings
    use_fp8: bool = False
    fp8_format: str = "E5M2"
    use_shadow: bool = False

    # Checkpointing
    checkpoint_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    max_checkpoints: int = 3

    # Stability thresholds
    nan_patience: int = 10
    bit_stall_threshold: float = 0.5
    overflow_threshold: float = 0.01
    dead_neuron_threshold: float = 1e-8
    dead_neuron_window: int = 100

    # Logging intervals
    log_interval: int = 1
    eval_interval: int = 100

    # W&B configuration
    project: str = "altgrad"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Reproducibility
    seed: int = 42


def save_config(config: TrainConfig, path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: TrainConfig instance to save
        path: File path for YAML output

    Example:
        >>> config = TrainConfig(n_layer=12)
        >>> save_config(config, 'experiment.yaml')
    """
    config_dict = asdict(config)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def load_config(path: str) -> TrainConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        TrainConfig instance with loaded values

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed

    Example:
        >>> config = load_config('experiment.yaml')
        >>> print(config.n_layer)
        12
    """
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainConfig(**config_dict)


__all__ = [
    "TrainConfig",
    "load_config",
    "save_config",
]
