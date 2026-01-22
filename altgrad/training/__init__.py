"""Training infrastructure for AltGrad experiments.

Provides complete training infrastructure including:
  - nanoGPT-style model architecture
  - Data loading and tokenization
  - Training configuration with YAML serialization
  - Gradient and stability metrics computation
  - FP32 shadow model for gradient comparison
  - Training loop orchestration
  - Checkpoint management with rotation
  - W&B logging and alerts

Submodules:
    model: GPT model architecture (nanoGPT-style)
    data: EurLex data preparation and batch loading
    config: TrainConfig dataclass and YAML I/O
    metrics: Gradient statistics and stability metrics
    shadow: FP32 shadow model for gradient comparison
    trainer: Training loop orchestration
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
from altgrad.training.model import GPT, GPTConfig
from altgrad.training.shadow import FP32ShadowModel
from altgrad.training.trainer import Trainer
from altgrad.training.format_runner import (
    FormatExperimentRunner,
    ExperimentResult,
    DiagnosticSnapshot,
    run_format_experiment,
)

__all__ = [
    # Model
    "GPT",
    "GPTConfig",
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
    # Shadow
    "FP32ShadowModel",
    # Trainer
    "Trainer",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
    # Callbacks
    "WandbTracker",
    # Format experiment runner
    "FormatExperimentRunner",
    "ExperimentResult",
    "DiagnosticSnapshot",
    "run_format_experiment",
]
