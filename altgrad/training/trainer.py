"""Training loop orchestration with quantization support.

Provides a Trainer class that integrates model, data, optimizer, gradient
scaling, checkpointing, and W&B logging into a coherent training loop.

Key features:
  - Mixed precision training (BF16/FP32)
  - Optional FP32 shadow model for gradient comparison
  - Optional bit-stall detection for FP8 training
  - Automatic checkpointing on intervals and anomalies
  - Gradient clipping and statistics logging
  - W&B integration with alerts

Example:
    >>> from altgrad.training import GPT, GPTConfig, TrainConfig, Trainer
    >>> model_config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
    >>> train_config = TrainConfig(batch_size=32, max_steps=1000)
    >>> model = GPT(model_config)
    >>> trainer = Trainer(train_config, model, "data/eurlex", "cuda")
    >>> trainer.train()
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from altgrad.training.config import TrainConfig
from altgrad.training.data import get_batch
from altgrad.training.metrics import compute_gradient_stats, compute_stability_metrics
from altgrad.training.checkpoint import CheckpointManager, load_checkpoint
from altgrad.training.shadow import FP32ShadowModel


class Trainer:
    """Training loop orchestrator for GPT models.

    Integrates all training components:
      - Model forward/backward with mixed precision
      - Optimizer with gradient clipping
      - Optional FP32 shadow for gradient comparison
      - Optional bit-stall detection for FP8
      - Checkpointing with rotation
      - W&B logging with alerts

    Attributes:
        config: Training configuration
        model: GPT model to train
        device: Training device (cpu/cuda/mps)
        optimizer: AdamW optimizer
        scaler: GradScaler for mixed precision
        checkpoint_manager: Handles checkpoint save/load/rotation
        tracker: W&B logging (if enabled)
        shadow: FP32 shadow model (if use_shadow)
        bit_stall_detector: BitStallDetector (if use_fp8)
        step: Current training step
        best_val_loss: Best validation loss seen

    Example:
        >>> trainer = Trainer(config, model, "data/eurlex", "cuda")
        >>> trainer.train(max_steps=1000)
    """

    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        data_dir: str,
        device: str,
    ):
        """Initialize trainer with all components.

        Args:
            config: Training configuration
            model: GPT model to train
            data_dir: Directory with train.bin/val.bin data files
            device: Training device (cpu/cuda/mps)
        """
        self.config = config
        self.model = model.to(device)
        self.data_dir = data_dir
        self.device = device

        # Determine device type for optimizer/autocast
        self.device_type = "cuda" if "cuda" in device else "cpu"

        # Configure optimizer
        self.optimizer = model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=config.learning_rate,
            betas=(0.9, 0.95),
            device_type=self.device_type,
        )

        # Gradient scaler for mixed precision
        # Use enabled=True only for CUDA (MPS/CPU don't support AMP scaler)
        self.scaler = GradScaler(enabled=(self.device_type == "cuda"))

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            max_checkpoints=config.max_checkpoints,
        )

        # W&B tracker (lazy import to allow mocking)
        self.tracker: Optional[Any] = None
        if config.project:  # Only initialize if project is set
            try:
                from altgrad.training.callbacks import WandbTracker
                self.tracker = WandbTracker(config)
            except ImportError:
                pass  # W&B not installed

        # FP32 shadow model for gradient comparison
        self.shadow: Optional[FP32ShadowModel] = None
        if config.use_shadow:
            self.shadow = FP32ShadowModel(model)

        # Bit-stall detector for FP8 training
        self.bit_stall_detector: Optional[Any] = None
        if config.use_fp8:
            from altgrad.quantization import BitStallDetector
            self.bit_stall_detector = BitStallDetector()

        # Training state
        self.step = 0
        self.best_val_loss = float("inf")

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Execute one training step.

        Forward pass, backward pass, optional shadow comparison,
        gradient clipping, and optimizer step.

        Args:
            x: Input token indices (batch, seq_len)
            y: Target token indices (batch, seq_len)

        Returns:
            Dictionary with training metrics:
              - loss: Training loss
              - perplexity: exp(loss)
              - grad_norm: Global gradient norm before clipping
              - grad_*: Gradient statistics
              - grad_cos_sim/*: Shadow gradient similarity (if use_shadow)
              - grad_snr/*: SNR comparison (if use_shadow)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass with autocast
        # Use bfloat16 for non-FP8, float32 for simulated FP8 quantization
        dtype = torch.float32 if self.config.use_fp8 else torch.bfloat16
        with torch.amp.autocast(device_type=self.device_type, dtype=dtype):
            logits, loss = self.model(x, y)

        metrics: Dict[str, float] = {
            "loss": loss.item(),
            "perplexity": math.exp(loss.item()) if loss.item() < 20 else float("inf"),
        }

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # Unscale for gradient stats and clipping
        self.scaler.unscale_(self.optimizer)

        # Shadow model forward/backward for gradient comparison
        if self.shadow is not None:
            # Sync weights to shadow (in case they diverged)
            self.shadow.sync_weights(self.model)
            # Run shadow forward/backward
            shadow_loss = self.shadow.forward_backward(x.to(self.device), y.to(self.device))
            metrics["shadow_loss"] = shadow_loss.item() if shadow_loss is not None else 0.0
            # Compute gradient similarity and SNR
            similarity_metrics = self.shadow.compute_gradient_similarity(self.model)
            metrics.update(similarity_metrics)

        # Gradient statistics
        grad_stats = compute_gradient_stats(self.model)
        metrics.update(grad_stats)

        # Global gradient norm (for logging before clipping)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        metrics["grad_norm"] = math.sqrt(total_norm)

        # Gradient clipping
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return metrics

    def eval_step(self) -> float:
        """Evaluate on validation set.

        Returns:
            Mean validation loss
        """
        self.model.eval()
        losses = []

        # Evaluate on multiple batches
        eval_iters = 10
        with torch.no_grad():
            for _ in range(eval_iters):
                x, y = get_batch(
                    "val",
                    self.data_dir,
                    self.config.block_size,
                    self.config.batch_size,
                    self.device,
                )
                dtype = torch.float32 if self.config.use_fp8 else torch.bfloat16
                with torch.amp.autocast(device_type=self.device_type, dtype=dtype):
                    _, loss = self.model(x, y)
                losses.append(loss.item())

        return sum(losses) / len(losses)

    def train(self, max_steps: Optional[int] = None) -> None:
        """Run main training loop.

        Args:
            max_steps: Override config.max_steps (useful for testing)
        """
        max_steps = max_steps or self.config.max_steps

        while self.step < max_steps:
            step_start = time.time()

            # Get training batch
            x, y = get_batch(
                "train",
                self.data_dir,
                self.config.block_size,
                self.config.batch_size,
                self.device,
            )

            # Execute training step
            metrics = self.train_step(x, y)

            # Compute throughput
            step_time = time.time() - step_start
            tokens_per_sec = self.config.batch_size * self.config.block_size / step_time
            metrics["throughput_tokens_sec"] = tokens_per_sec
            metrics["step_time_ms"] = step_time * 1000

            # Stability metrics
            stability = compute_stability_metrics(self.model, self.bit_stall_detector)
            metrics.update(stability)

            # Check for alerts and handle actions
            action = "continue"
            if self.tracker is not None:
                action = self.tracker.check_alerts(self.step, metrics, self.config)

            if action == "save_checkpoint":
                self.checkpoint_manager.save_on_anomaly(
                    self.step,
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.config,
                )
            elif action == "stop":
                print(f"Training stopped at step {self.step} due to alerts")
                break

            # Evaluation on interval
            val_loss = None
            if self.step > 0 and self.step % self.config.eval_interval == 0:
                val_loss = self.eval_step()
                metrics["val_loss"] = val_loss
                metrics["val_perplexity"] = math.exp(val_loss) if val_loss < 20 else float("inf")

                # Track best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            # Log metrics
            self._log_metrics(self.step, metrics, val_loss)

            # Checkpoint on interval
            if self.step > 0 and self.step % self.config.checkpoint_interval == 0:
                current_val = val_loss if val_loss is not None else self.best_val_loss
                self.checkpoint_manager.save(
                    self.step,
                    self.model,
                    self.optimizer,
                    self.scaler,
                    self.config,
                    current_val,
                )

            self.step += 1

        # Final checkpoint
        if self.step > 0:
            self.checkpoint_manager.save(
                self.step,
                self.model,
                self.optimizer,
                self.scaler,
                self.config,
                self.best_val_loss,
            )

        # Finish W&B run
        if self.tracker is not None:
            self.tracker.finish()

    def _log_metrics(
        self,
        step: int,
        train_metrics: Dict[str, float],
        val_loss: Optional[float] = None,
    ) -> None:
        """Log metrics to W&B.

        Args:
            step: Current training step
            train_metrics: Metrics from train_step
            val_loss: Validation loss (if evaluated this step)
        """
        if self.step % self.config.log_interval != 0:
            return

        # Learning rate (from first param group)
        train_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Log to W&B
        if self.tracker is not None:
            self.tracker.log_step(step, train_metrics)
        else:
            # Console logging
            loss = train_metrics.get("loss", 0)
            ppl = train_metrics.get("perplexity", 0)
            print(f"Step {step}: loss={loss:.4f}, ppl={ppl:.2f}")

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        step, config_dict, quant_state = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scaler,
        )
        self.step = step + 1  # Resume from next step
        print(f"Resumed from step {step}")


__all__ = [
    "Trainer",
]
