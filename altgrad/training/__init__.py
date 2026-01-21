"""Training infrastructure for AltGrad experiments.

Provides complete training infrastructure including:
  - Data loading and tokenization
  - Training configuration with YAML serialization
  - Gradient and stability metrics computation
  - Checkpoint management with rotation
  - W&B logging and alerts

Submodules:
    data: EurLex data preparation and batch loading
    config: TrainConfig dataclass and YAML I/O
    metrics: Gradient statistics and stability metrics
    checkpoint: Save/load with full state restoration
    callbacks: W&B tracking with alerts
"""

from altgrad.training.data import prepare_eurlex, get_batch
from altgrad.training.config import TrainConfig, load_config, save_config
from altgrad.training.metrics import (
    compute_gradient_stats,
    compute_stability_metrics,
    gradient_cosine_similarity,
)
from altgrad.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointManager,
)
from altgrad.training.callbacks import WandbTracker

__all__ = [
    # Data
    "prepare_eurlex",
    "get_batch",
    # Config
    "TrainConfig",
    "load_config",
    "save_config",
    # Metrics
    "compute_gradient_stats",
    "compute_stability_metrics",
    "gradient_cosine_similarity",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
    # Callbacks
    "WandbTracker",
]
