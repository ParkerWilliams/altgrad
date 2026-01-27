"""Training infrastructure for AltGrad experiments.

Provides complete training infrastructure including:
  - nanoGPT-style model architecture (legacy LM)
  - Classification model with transformer encoder
  - Data loading for both LM and classification
  - Training configuration with YAML serialization
  - Gradient and stability metrics computation
  - Classification metrics (F1, precision, recall, AUC)
  - Rank collapse monitoring
  - FP32 shadow model for gradient comparison
  - Training loop orchestration
  - Checkpoint management with rotation
  - W&B logging and alerts

Submodules:
    model: GPT model architecture (nanoGPT-style, legacy)
    classifier: Transformer classifier for multi-label classification
    data: EurLex data preparation for LM (legacy)
    classification_data: EUR-Lex multi-label classification data
    classification_metrics: F1, precision, recall, AUC metrics
    classification_trainer: Classification training loop
    config: TrainConfig dataclass and YAML I/O
    metrics: Gradient statistics, stability metrics, rank stats
    shadow: FP32 shadow model for gradient comparison
    trainer: LM training loop (legacy)
    checkpoint: Save/load with full state restoration
    callbacks: W&B tracking with alerts
"""

from altgrad.training.data import prepare_eurlex, get_batch
from altgrad.training.config import TrainConfig, load_config, save_config
from altgrad.training.metrics import (
    compute_gradient_stats,
    compute_stability_metrics,
    gradient_cosine_similarity,
    compute_rank_stats,
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
from altgrad.training.optimizer import ManifoldAdamW, GridOptim

# Classification modules
from altgrad.training.classifier import TransformerClassifier, ClassifierConfig
from altgrad.training.classification_data import (
    EURLexDataset,
    create_dataloaders,
)
from altgrad.training.classification_metrics import (
    ClassificationMetrics,
    MetricsComputer,
    compute_throughput_metrics,
)
from altgrad.training.classification_trainer import (
    ClassificationTrainer,
    ClassificationTrainConfig,
)

__all__ = [
    # Model (legacy LM)
    "GPT",
    "GPTConfig",
    # Classification model
    "TransformerClassifier",
    "ClassifierConfig",
    # Data (legacy LM)
    "prepare_eurlex",
    "get_batch",
    # Classification data
    "EURLexDataset",
    "create_dataloaders",
    # Config
    "TrainConfig",
    "load_config",
    "save_config",
    # Gradient/stability metrics
    "compute_gradient_stats",
    "compute_stability_metrics",
    "gradient_cosine_similarity",
    "compute_rank_stats",
    # Classification metrics
    "ClassificationMetrics",
    "MetricsComputer",
    "compute_throughput_metrics",
    # Shadow
    "FP32ShadowModel",
    # Trainer (legacy LM)
    "Trainer",
    # Classification trainer
    "ClassificationTrainer",
    "ClassificationTrainConfig",
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
    # Optimizer
    "ManifoldAdamW",
    "GridOptim",
]
