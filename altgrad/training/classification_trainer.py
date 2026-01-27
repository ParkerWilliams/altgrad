"""Classification training loop with full metrics integration.

Trainer for multi-label document classification that integrates:
- FP8 quantization with STE
- ManifoldAdamW optimizer
- Rank collapse monitoring (the key metric!)
- Flip metrics and stall ratios
- All classification metrics (F1, precision, recall, AUC, etc.)
- Throughput tracking
- W&B logging
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from altgrad.training.classifier import TransformerClassifier, ClassifierConfig
from altgrad.training.classification_metrics import (
    MetricsComputer,
    ClassificationMetrics,
    compute_throughput_metrics,
)
from altgrad.training.metrics import compute_gradient_stats, compute_rank_stats
from altgrad.training.optimizer import ManifoldAdamW
from altgrad.quantization import (
    quantize,
    compute_scale,
    AmaxHistory,
    FORMAT_REGISTRY,
    FP8Format,
)
from altgrad.quantization.rank_health import RankHealthMonitor
from altgrad.quantization.flip_metrics import WeightFlipTracker


@dataclass
class ClassificationTrainConfig:
    """Configuration for classification training."""

    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_steps: int = 5000
    batch_size: int = 16
    max_length: int = 512
    grad_clip: float = 1.0

    # Evaluation
    eval_interval: int = 500
    log_interval: int = 50

    # FP8 quantization
    use_fp8: bool = False
    fp8_format: str = "E5M2"
    quantize_encoder: bool = True
    quantize_classifier: bool = False

    # ManifoldAdamW
    use_manifold_aware: bool = False
    manifold_mantissa_bits: int = 2
    manifold_max_stiffness: float = 1e6

    # Rank monitoring (THE KEY METRIC)
    rank_log_interval: int = 100
    rank_warn_threshold: float = 0.3  # Warn if rank drops 30%

    # Flip tracking
    track_flips: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000

    # W&B
    project: Optional[str] = None
    run_name: Optional[str] = None


class ClassificationTrainer:
    """Trainer for multi-label classification with full diagnostics.

    Integrates all monitoring components:
    - Rank health monitoring for collapse detection
    - Flip tracking for quantization dynamics
    - All classification metrics
    - Gradient statistics
    - Throughput tracking

    This is the trainer that should have been built from the start.

    Example:
        >>> config = ClassificationTrainConfig(use_fp8=True, fp8_format="E5M2")
        >>> model = TransformerClassifier(ClassifierConfig(num_labels=1000))
        >>> trainer = ClassificationTrainer(config, model, train_loader, val_loader, "cuda")
        >>> trainer.train()
    """

    def __init__(
        self,
        config: ClassificationTrainConfig,
        model: TransformerClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            model: Classification model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Training device
        """
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        if config.use_manifold_aware:
            self.optimizer = self._configure_manifold_optimizer()
        else:
            self.optimizer = model.configure_optimizers(
                weight_decay=config.weight_decay,
                learning_rate=config.learning_rate,
            )

        # FP8 setup
        self.fp8_format: Optional[FP8Format] = None
        self.amax_histories: Dict[str, AmaxHistory] = {}

        if config.use_fp8:
            self.fp8_format = FORMAT_REGISTRY[config.fp8_format]
            for name, param in model.named_parameters():
                if model.should_quantize(name) and param.dim() >= 2:
                    self.amax_histories[name] = AmaxHistory(history_len=16)

        # RANK MONITORING - THE KEY METRIC
        self.rank_monitor = RankHealthMonitor(
            log_interval=config.rank_log_interval,
            warn_threshold=config.rank_warn_threshold,
        )

        # Flip tracking
        self.flip_tracker: Optional[WeightFlipTracker] = None
        if config.track_flips and config.use_fp8:
            self.flip_tracker = WeightFlipTracker()

        # Metrics computer
        self.metrics_computer = MetricsComputer(
            num_labels=model.config.num_labels,
            threshold=0.5,
        )

        # W&B tracker
        self.tracker: Optional[Any] = None
        if config.project:
            try:
                import wandb
                wandb.init(
                    project=config.project,
                    name=config.run_name,
                    config={
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "max_steps": config.max_steps,
                        "fp8_format": config.fp8_format if config.use_fp8 else "none",
                        "use_manifold_aware": config.use_manifold_aware,
                        "rank_warn_threshold": config.rank_warn_threshold,
                    },
                )
                self.tracker = wandb
            except ImportError:
                pass

        # Training state
        self.step = 0
        self.best_f1_micro = 0.0

        # Accumulated metrics for logging
        self.train_loss_accum = 0.0
        self.train_steps_accum = 0

    def _configure_manifold_optimizer(self) -> ManifoldAdamW:
        """Configure ManifoldAdamW optimizer."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return ManifoldAdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            manifold_aware=True,
            mantissa_bits=self.config.manifold_mantissa_bits,
            max_stiffness=self.config.manifold_max_stiffness,
        )

    @contextmanager
    def _quantized_forward_context(self):
        """Context manager for simulated FP8 forward pass."""
        if not self.config.use_fp8:
            yield
            return

        original_weights = {}

        try:
            for name, param in self.model.named_parameters():
                if name in self.amax_histories:
                    original_weights[name] = param.data.clone()

                    self.amax_histories[name].update(param.data)
                    amax = self.amax_histories[name].get_amax()
                    scale = compute_scale(amax, self.fp8_format)
                    scale_tensor = torch.tensor(scale, device=param.device, dtype=param.dtype)

                    quantized = quantize(param.data, self.fp8_format, scale_tensor)
                    param.data.copy_(quantized)

            yield

        finally:
            for name, orig in original_weights.items():
                parts = name.split(".")
                module = self.model
                for part in parts[:-1]:
                    module = getattr(module, part)
                param = getattr(module, parts[-1])
                param.data.copy_(orig)

    def _snapshot_flips_pre_step(self) -> None:
        """Capture quantized weight state before optimizer step."""
        if self.flip_tracker is None or not self.config.use_fp8:
            return

        for name, param in self.model.named_parameters():
            if name in self.amax_histories:
                amax = self.amax_histories[name].get_amax()
                scale = compute_scale(amax, self.fp8_format)
                scale_tensor = torch.tensor(scale, device=param.device, dtype=param.dtype)

                self.flip_tracker.snapshot_pre_step(
                    name, param.data, self.fp8_format, scale_tensor, param.grad
                )

    def _compute_flips_post_step(self) -> Dict[str, float]:
        """Compute weight flips after optimizer step."""
        if self.flip_tracker is None or not self.config.use_fp8:
            return {}

        total_flips = 0
        for name, param in self.model.named_parameters():
            if name in self.amax_histories:
                amax = self.amax_histories[name].get_amax()
                scale = compute_scale(amax, self.fp8_format)
                scale_tensor = torch.tensor(scale, device=param.device, dtype=param.dtype)

                flips = self.flip_tracker.compute_flips_post_step(
                    name, param.data, self.fp8_format, scale_tensor
                )
                total_flips += flips

        flip_rates = self.flip_tracker.get_flip_rates()
        stall_ratios = self.flip_tracker.get_stall_ratios()

        metrics = {
            "flips/total": total_flips,
            "flips/mean_rate": sum(flip_rates.values()) / max(len(flip_rates), 1),
            "stall/mean_ratio": sum(stall_ratios.values()) / max(len(stall_ratios), 1),
        }

        return metrics

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Execute one training step.

        Returns:
            Dictionary with all training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward with optional FP8 quantization
        with self._quantized_forward_context():
            logits, loss = self.model(input_ids, attention_mask, labels)

        metrics: Dict[str, float] = {"loss": loss.item()}

        # Backward
        loss.backward()

        # Gradient stats
        grad_stats = compute_gradient_stats(self.model)
        metrics.update(grad_stats)

        # Gradient clipping
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Snapshot flips before optimizer step
        self._snapshot_flips_pre_step()

        # Optimizer step
        self.optimizer.step()

        # Compute flips after optimizer step
        flip_metrics = self._compute_flips_post_step()
        metrics.update(flip_metrics)

        # RANK MONITORING - computed at intervals
        if self.step % self.config.rank_log_interval == 0:
            rank_stats = compute_rank_stats(self.model)
            metrics.update(rank_stats)

            # Check for rank collapse warnings
            layer_ranks = self.rank_monitor.compute_layer_ranks(self.model)
            warnings = self.rank_monitor.check_warnings(layer_ranks)

            for warning in warnings:
                print(f"[RANK COLLAPSE WARNING] {warning}")
                if self.tracker:
                    self.tracker.alert(
                        title="Rank Collapse Warning",
                        text=warning,
                        level="warning",
                    )

        return metrics

    def evaluate(self) -> ClassificationMetrics:
        """Evaluate on validation set.

        Returns:
            ClassificationMetrics with all computed metrics
        """
        self.model.eval()
        self.metrics_computer.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                with self._quantized_forward_context():
                    logits, _ = self.model(input_ids, attention_mask)

                self.metrics_computer.update(logits.cpu(), labels)

        return self.metrics_computer.compute()

    def train(self) -> None:
        """Run main training loop."""
        train_iter = iter(self.train_loader)
        step_start = time.time()

        while self.step < self.config.max_steps:
            # Get batch (with wraparound)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Training step
            metrics = self.train_step(batch)

            # Throughput
            step_time = time.time() - step_start
            throughput = compute_throughput_metrics(
                batch_size=self.config.batch_size,
                seq_length=self.config.max_length,
                step_time_seconds=step_time,
            )
            metrics.update(throughput)

            # Accumulate loss
            self.train_loss_accum += metrics["loss"]
            self.train_steps_accum += 1

            # Logging
            if self.step % self.config.log_interval == 0:
                avg_loss = self.train_loss_accum / max(self.train_steps_accum, 1)
                self._log_train_metrics(metrics, avg_loss)
                self.train_loss_accum = 0.0
                self.train_steps_accum = 0

            # Evaluation
            if self.step > 0 and self.step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                self._log_eval_metrics(eval_metrics)

                if eval_metrics.f1_micro > self.best_f1_micro:
                    self.best_f1_micro = eval_metrics.f1_micro
                    self._save_checkpoint("best")

            # Checkpoint
            if self.step > 0 and self.step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(f"step_{self.step}")

            self.step += 1
            step_start = time.time()

        # Final evaluation
        final_metrics = self.evaluate()
        self._log_eval_metrics(final_metrics)
        self._save_checkpoint("final")

        if self.tracker:
            self.tracker.finish()

    def _log_train_metrics(self, metrics: Dict[str, float], avg_loss: float) -> None:
        """Log training metrics."""
        # Console
        log_parts = [
            f"Step {self.step}",
            f"loss={avg_loss:.4f}",
            f"samples/s={metrics.get('throughput/samples_per_sec', 0):.1f}",
        ]

        if "stable_rank/mean" in metrics:
            log_parts.append(f"stable_rank={metrics['stable_rank/mean']:.2f}")
            log_parts.append(f"effective_rank={metrics['effective_rank/mean']:.2f}")

        if "stall/mean_ratio" in metrics:
            log_parts.append(f"stall_ratio={metrics['stall/mean_ratio']:.2%}")

        if "flips/total" in metrics:
            log_parts.append(f"flips={int(metrics['flips/total'])}")

        print(" | ".join(log_parts))

        # W&B
        if self.tracker:
            self.tracker.log({"train/" + k: v for k, v in metrics.items()}, step=self.step)
            self.tracker.log({"train/avg_loss": avg_loss}, step=self.step)

    def _log_eval_metrics(self, metrics: ClassificationMetrics) -> None:
        """Log evaluation metrics."""
        print(f"\n=== Evaluation at step {self.step} ===")
        print(f"F1 micro:  {metrics.f1_micro:.4f}")
        print(f"F1 macro:  {metrics.f1_macro:.4f}")
        print(f"Precision: {metrics.precision_micro:.4f}")
        print(f"Recall:    {metrics.recall_micro:.4f}")
        print(f"Hamming:   {metrics.hamming_loss:.4f}")
        print(f"Subset:    {metrics.subset_accuracy:.4f}")
        print(f"ROC-AUC:   {metrics.roc_auc_micro:.4f}")
        print(f"PR-AUC:    {metrics.pr_auc_micro:.4f}")
        print("=" * 40 + "\n")

        if self.tracker:
            eval_dict = {"eval/" + k: v for k, v in metrics.to_dict().items()}
            self.tracker.log(eval_dict, step=self.step)

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1_micro": self.best_f1_micro,
            "config": self.config,
        }, path)

        print(f"Saved checkpoint: {path}")


__all__ = [
    "ClassificationTrainConfig",
    "ClassificationTrainer",
]
