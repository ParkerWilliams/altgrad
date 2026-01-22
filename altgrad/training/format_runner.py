"""Self-contained format experiment runner for Phase 4 testing.

Runs complete training experiments for exotic FP8 formats with:
- Stability interventions (PartitionRelativeClipper, EmergencyMantissaShift)
- Advanced diagnostics (stiffness, grid alignment, ULP statistics)
- Failure mode report generation (markdown artifact on collapse)
- Checkpoint on NaN detection

Example:
    >>> from altgrad.training import FormatExperimentRunner
    >>> runner = FormatExperimentRunner(config, model, "data/eurlex", "cuda")
    >>> result = runner.run()
    >>> print(result.final_loss)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler

from altgrad.training.config import TrainConfig
from altgrad.training.data import get_batch
from altgrad.training.metrics import compute_gradient_stats, compute_stability_metrics
from altgrad.training.checkpoint import CheckpointManager, save_checkpoint
from altgrad.training.shadow import FP32ShadowModel
from altgrad.training.trainer import Trainer

# Quantization imports
from altgrad.quantization import (
    quantize,
    compute_scale,
    AmaxHistory,
    BitStallDetector,
    FORMAT_REGISTRY,
)

# Stability interventions from Phase 4 Wave 1
from altgrad.quantization.stability import (
    PartitionRelativeClipper,
    EmergencyMantissaShift,
)

# Advanced diagnostics from Phase 4 Wave 1
from altgrad.quantization.advanced_diagnostics import (
    compute_stiffness_field,
    grid_alignment_statistics,
    ulp_statistics,
    gradient_stiffness_correlation,
)


@dataclass
class DiagnosticSnapshot:
    """Snapshot of advanced diagnostics at a training step."""

    step: int
    stiffness_mean: float = 0.0
    stiffness_max: float = 0.0
    stiffness_std: float = 0.0
    grid_error_mean: float = 0.0
    grid_error_max: float = 0.0
    on_grid_frac: float = 0.0
    ulp_mean: float = 0.0
    ulp_median: float = 0.0
    ulp_zero_frac: float = 0.0
    grad_stiff_correlation: float = 0.0
    grad_below_stiffness_frac: float = 0.0


@dataclass
class ExperimentResult:
    """Complete results from a format experiment run.

    Attributes:
        format_name: FP8 format used (e.g., "E7M0", "E3M4")
        completed: Whether experiment ran to completion
        steps_completed: Number of steps completed
        max_steps: Maximum steps configured
        final_loss: Final training loss (or loss at collapse)
        best_loss: Best training loss achieved
        collapse_step: Step at which collapse occurred (None if completed)
        collapse_reason: Why experiment collapsed (None if completed)
        format_shifted: Whether emergency format shift occurred
        shifted_to: Format shifted to (None if no shift)
        diagnostics: List of diagnostic snapshots
        failure_report_path: Path to failure report (None if completed)
        duration_seconds: Total experiment duration
        metrics_history: Full metrics history for analysis
    """

    format_name: str
    completed: bool
    steps_completed: int
    max_steps: int
    final_loss: float
    best_loss: float
    collapse_step: Optional[int] = None
    collapse_reason: Optional[str] = None
    format_shifted: bool = False
    shifted_to: Optional[str] = None
    diagnostics: List[DiagnosticSnapshot] = field(default_factory=list)
    failure_report_path: Optional[str] = None
    duration_seconds: float = 0.0
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)


class FormatExperimentRunner:
    """Self-contained experiment runner for FP8 format testing.

    Creates its own Trainer and runs a complete training loop with:
    - PartitionRelativeClipper for format-aware gradient clipping
    - EmergencyMantissaShift for fallback on training collapse
    - Advanced diagnostics logging to W&B
    - Failure mode report generation on collapse

    Attributes:
        config: Training configuration
        model: GPT model to train
        data_dir: Directory with train.bin/val.bin
        device: Training device
        trainer: Underlying Trainer instance

    Example:
        >>> runner = FormatExperimentRunner(config, model, "data/eurlex", "cuda")
        >>> result = runner.run()
    """

    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        data_dir: str,
        device: str,
    ):
        """Initialize experiment runner.

        Args:
            config: Training configuration with FP8 settings
            model: GPT model to train
            data_dir: Directory with train.bin/val.bin data files
            device: Training device (cpu/cuda/mps)
        """
        self.config = config
        self.model = model
        self.data_dir = data_dir
        self.device = device
        self.device_type = "cuda" if "cuda" in device else "cpu"

        # Create underlying Trainer
        self.trainer = Trainer(config, model, data_dir, device)

        # Initialize stability interventions
        self.clipper: Optional[PartitionRelativeClipper] = None
        self.shifter: Optional[EmergencyMantissaShift] = None
        self.current_format = config.fp8_format
        self.format_shifted = False
        self.shifted_to: Optional[str] = None

        if config.use_fp8:
            fp8_format = FORMAT_REGISTRY[config.fp8_format]

            if config.enable_partition_clipping:
                self.clipper = PartitionRelativeClipper(
                    format=fp8_format,
                    base_clip=config.partition_clip_base,
                    overflow_threshold=config.overflow_threshold,
                )

            if config.enable_emergency_shift:
                self.shifter = EmergencyMantissaShift(
                    nan_patience=config.emergency_shift_nan_patience,
                    stall_threshold=config.emergency_shift_stall_threshold,
                )

        # Tracking
        self.diagnostics: List[DiagnosticSnapshot] = []
        self.metrics_history: Dict[str, List[float]] = {
            "loss": [],
            "perplexity": [],
            "grad_norm": [],
            "bit_stall_rate": [],
            "overflow_rate": [],
        }
        self.best_loss = float("inf")
        self.collapse_step: Optional[int] = None
        self.collapse_reason: Optional[str] = None
        self.start_time: Optional[float] = None

        # Store weights before each step for ULP diagnostics
        self._prev_weights: Dict[str, torch.Tensor] = {}

    def _store_weights_snapshot(self) -> None:
        """Store current weights for ULP computation."""
        self._prev_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                self._prev_weights[name] = param.data.clone()

    def _collect_diagnostics(self, step: int, metrics: Dict[str, float]) -> DiagnosticSnapshot:
        """Collect advanced diagnostic metrics.

        Args:
            step: Current training step
            metrics: Metrics from train_step

        Returns:
            DiagnosticSnapshot with all diagnostic measurements
        """
        snapshot = DiagnosticSnapshot(step=step)

        if not self.config.use_fp8:
            return snapshot

        fp8_format = FORMAT_REGISTRY[self.current_format]
        mantissa_bits = fp8_format.mantissa_bits

        # Aggregate diagnostics across all tracked parameters
        all_stiffness = []
        all_grid_errors = []
        all_ulp_dists = []
        all_grad_below_stiff = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.dim() < 2:
                continue

            # Stiffness field (DIAG-01)
            if self.config.log_stiffness:
                stiffness = compute_stiffness_field(param.data, mantissa_bits)
                valid_stiffness = stiffness[~torch.isnan(stiffness)]
                if len(valid_stiffness) > 0:
                    all_stiffness.append(valid_stiffness)

            # Grid alignment (DIAG-02)
            if self.config.log_grid_alignment:
                if name in self.trainer.amax_histories:
                    amax = self.trainer.amax_histories[name].get_amax()
                    scale = compute_scale(amax, fp8_format)
                    scale_tensor = torch.tensor(scale, device=param.device, dtype=param.dtype)
                    grid_stats = grid_alignment_statistics(param.data, fp8_format, scale_tensor)
                    all_grid_errors.append(grid_stats["grid_error_mean"])

            # ULP statistics (DIAG-04)
            if self.config.log_ulp and name in self._prev_weights:
                ulp_stats = ulp_statistics(self._prev_weights[name], param.data)
                all_ulp_dists.append(ulp_stats["ulp_mean"])

            # Gradient-stiffness correlation (DIAG-03)
            if param.grad is not None:
                corr_stats = gradient_stiffness_correlation(
                    param.data, param.grad.data, mantissa_bits
                )
                all_grad_below_stiff.append(corr_stats["grad_below_stiffness_frac"])

        # Aggregate into snapshot
        if all_stiffness:
            combined = torch.cat(all_stiffness)
            snapshot.stiffness_mean = combined.mean().item()
            snapshot.stiffness_max = combined.max().item()
            snapshot.stiffness_std = combined.std().item()

        if all_grid_errors:
            snapshot.grid_error_mean = sum(all_grid_errors) / len(all_grid_errors)

        if all_ulp_dists:
            snapshot.ulp_mean = sum(all_ulp_dists) / len(all_ulp_dists)

        if all_grad_below_stiff:
            snapshot.grad_below_stiffness_frac = sum(all_grad_below_stiff) / len(all_grad_below_stiff)

        return snapshot

    def _log_diagnostics_to_wandb(self, step: int, snapshot: DiagnosticSnapshot) -> None:
        """Log diagnostic snapshot to W&B.

        Args:
            step: Current training step
            snapshot: Diagnostic measurements
        """
        if self.trainer.tracker is None:
            return

        diag_metrics = {}

        if self.config.log_stiffness:
            diag_metrics["diagnostics/stiffness_mean"] = snapshot.stiffness_mean
            diag_metrics["diagnostics/stiffness_max"] = snapshot.stiffness_max
            diag_metrics["diagnostics/stiffness_std"] = snapshot.stiffness_std

        if self.config.log_grid_alignment:
            diag_metrics["diagnostics/grid_error_mean"] = snapshot.grid_error_mean

        if self.config.log_ulp:
            diag_metrics["diagnostics/ulp_mean"] = snapshot.ulp_mean

        diag_metrics["diagnostics/grad_below_stiffness_frac"] = snapshot.grad_below_stiffness_frac

        if diag_metrics:
            self.trainer.tracker.log_step(step, diag_metrics)

    def _check_collapse(self, step: int, metrics: Dict[str, float]) -> bool:
        """Check if training has collapsed.

        Args:
            step: Current training step
            metrics: Metrics from train_step

        Returns:
            True if training should stop due to collapse
        """
        loss = metrics.get("loss", 0)

        # Check for NaN loss
        if math.isnan(loss) or math.isinf(loss):
            self.collapse_step = step
            self.collapse_reason = "NaN/Inf loss"
            return True

        # Check for bit-stall rate exceeding threshold
        stall_rate = metrics.get("quantization/bit_stall_rate", 0)
        if stall_rate > self.config.bit_stall_threshold:
            # Not immediate collapse, but track
            pass

        return False

    def _handle_stability_interventions(
        self, step: int, metrics: Dict[str, float]
    ) -> Optional[str]:
        """Apply stability interventions and check for format shift.

        Args:
            step: Current training step
            metrics: Metrics from train_step

        Returns:
            New format name if shift occurred, None otherwise
        """
        if not self.config.use_fp8:
            return None

        loss = metrics.get("loss", 0)
        stall_rate = metrics.get("quantization/bit_stall_rate", 0)
        overflow_rate = metrics.get("quantization/overflow_rate", 0)
        has_nan = math.isnan(loss) or math.isinf(loss)

        # Apply partition-relative clipping (STAB-05)
        if self.clipper is not None and not has_nan:
            clipped = self.clipper.clip_if_needed(self.model, overflow_rate)
            if clipped:
                print(f"Step {step}: Applied partition-relative clipping (overflow={overflow_rate:.2%})")

        # Check for emergency format shift (STAB-06)
        if self.shifter is not None:
            new_format = self.shifter.check_and_shift(
                self.current_format, has_nan, stall_rate
            )
            if new_format is not None:
                print(f"Step {step}: Emergency shift from {self.current_format} to {new_format}")
                self.format_shifted = True
                self.shifted_to = new_format
                self._apply_format_shift(new_format)
                return new_format

        return None

    def _apply_format_shift(self, new_format: str) -> None:
        """Apply format shift to trainer.

        Args:
            new_format: New FP8 format name
        """
        self.current_format = new_format
        self.trainer.fp8_format = FORMAT_REGISTRY[new_format]
        self.config.fp8_format = new_format

        # Update clipper with new format
        if self.clipper is not None:
            self.clipper = PartitionRelativeClipper(
                format=FORMAT_REGISTRY[new_format],
                base_clip=self.config.partition_clip_base,
                overflow_threshold=self.config.overflow_threshold,
            )

    def _generate_failure_report(
        self, result: ExperimentResult, final_metrics: Dict[str, float]
    ) -> str:
        """Generate markdown failure report artifact.

        Args:
            result: Experiment result with collapse info
            final_metrics: Final metrics at collapse

        Returns:
            Path to generated report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.config.checkpoint_dir) / "failure_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{self.config.fp8_format}_{timestamp}_failure_report.md"

        # Compute gradient sparsity (fraction of near-zero gradients)
        total_params = 0
        zero_grads = 0
        zero_update_layers: List[str] = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                total_params += grad.numel()
                near_zero = (grad.abs() < 1e-8).sum().item()
                zero_grads += near_zero

                # Track layers with >90% zero gradient
                if near_zero / grad.numel() > 0.9:
                    zero_update_layers.append(name)

        grad_sparsity = zero_grads / max(total_params, 1)

        # Get last N diagnostics for trend analysis
        last_diagnostics = result.diagnostics[-5:] if len(result.diagnostics) > 5 else result.diagnostics

        # Build report content
        report = f"""# Failure Report: {self.config.fp8_format} Experiment

## Summary

| Metric | Value |
|--------|-------|
| Format | {self.config.fp8_format} |
| Collapse Step | {result.collapse_step} |
| Collapse Reason | {result.collapse_reason} |
| Final Loss | {result.final_loss:.6f} |
| Best Loss | {result.best_loss:.6f} |
| Total Steps | {result.steps_completed} / {result.max_steps} |
| Duration | {result.duration_seconds:.1f}s |

## Collapse Analysis

### Gradient Sparsity

Fraction of near-zero gradients at collapse: **{grad_sparsity:.2%}**

This indicates {"severe gradient vanishing" if grad_sparsity > 0.5 else "moderate gradient flow"}.

### Zero-Update Regions

Layers with >90% zero gradients at collapse:

"""
        if zero_update_layers:
            for layer in zero_update_layers[:10]:  # Limit to first 10
                report += f"- `{layer}`\n"
            if len(zero_update_layers) > 10:
                report += f"- ... and {len(zero_update_layers) - 10} more\n"
        else:
            report += "- None (gradients were flowing)\n"

        report += f"""
### Quantization Metrics at Collapse

| Metric | Value |
|--------|-------|
| Bit-Stall Rate | {final_metrics.get('quantization/bit_stall_rate', 0):.2%} |
| Overflow Rate | {final_metrics.get('quantization/overflow_rate', 0):.2%} |
| Underflow Rate | {final_metrics.get('quantization/underflow_rate', 0):.2%} |

### Diagnostic Trend (Last 5 Measurements)

"""
        if last_diagnostics:
            report += "| Step | Stiffness Mean | Grid Error | ULP Mean | Grad < Stiff |\n"
            report += "|------|----------------|------------|----------|---------------|\n"
            for diag in last_diagnostics:
                report += f"| {diag.step} | {diag.stiffness_mean:.6f} | {diag.grid_error_mean:.6f} | {diag.ulp_mean:.2f} | {diag.grad_below_stiffness_frac:.2%} |\n"
        else:
            report += "No diagnostic snapshots collected.\n"

        report += f"""
## Configuration

```yaml
fp8_format: {self.config.fp8_format}
max_steps: {self.config.max_steps}
batch_size: {self.config.batch_size}
learning_rate: {self.config.learning_rate}
grad_clip: {self.config.grad_clip}
enable_partition_clipping: {self.config.enable_partition_clipping}
enable_emergency_shift: {self.config.enable_emergency_shift}
seed: {self.config.seed}
```

## Metrics History

### Loss Trend

"""
        # Add loss trend (sampled)
        losses = result.metrics_history.get("loss", [])
        if losses:
            sample_interval = max(1, len(losses) // 20)  # ~20 samples
            report += "| Step | Loss |\n|------|------|\n"
            for i in range(0, len(losses), sample_interval):
                report += f"| {i} | {losses[i]:.6f} |\n"
            if len(losses) - 1 > 0 and (len(losses) - 1) % sample_interval != 0:
                report += f"| {len(losses)-1} | {losses[-1]:.6f} |\n"

        report += f"""
## Recommendations

Based on the failure analysis:

"""
        # Add recommendations based on failure mode
        if grad_sparsity > 0.7:
            report += """1. **High gradient sparsity detected**: Consider using a format with more mantissa bits (E3M4 or E5M2) to preserve gradient precision.
2. **Learning rate**: May need reduction to prevent numerical instability.
"""
        elif "NaN" in str(result.collapse_reason):
            report += """1. **NaN collapse**: Likely numerical overflow. Enable partition-relative clipping.
2. **Consider emergency shift**: Enable automatic fallback to higher-precision format.
"""
        else:
            report += """1. **General instability**: Review quantization scale factors.
2. **Try longer warmup**: Increase warmup_steps to allow model adaptation.
"""

        report += f"""
---
*Generated by FormatExperimentRunner at {datetime.now().isoformat()}*
"""

        # Write report
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Failure report written to: {report_path}")
        return str(report_path)

    def run(self, max_steps: Optional[int] = None) -> ExperimentResult:
        """Run the complete format experiment.

        Args:
            max_steps: Override config.max_steps (useful for testing)

        Returns:
            ExperimentResult with all metrics and analysis
        """
        max_steps = max_steps or self.config.max_steps
        self.start_time = time.time()

        print(f"Starting {self.config.fp8_format} format experiment")
        print(f"  Max steps: {max_steps}")
        print(f"  Partition clipping: {self.config.enable_partition_clipping}")
        print(f"  Emergency shift: {self.config.enable_emergency_shift}")
        print(f"  Diagnostic interval: {self.config.diagnostic_interval}")

        step = 0
        final_metrics: Dict[str, float] = {}

        while step < max_steps:
            # Store weights for ULP diagnostics
            if self.config.log_ulp and step % self.config.diagnostic_interval == 0:
                self._store_weights_snapshot()

            # Get training batch
            x, y = get_batch(
                "train",
                self.data_dir,
                self.config.block_size,
                self.config.batch_size,
                self.device,
            )

            # Execute training step
            try:
                metrics = self.trainer.train_step(x, y)
            except RuntimeError as e:
                # Catch CUDA errors, NaN-related crashes
                self.collapse_step = step
                self.collapse_reason = f"Runtime error: {str(e)[:100]}"
                break

            final_metrics = metrics

            # Track metrics history
            self.metrics_history["loss"].append(metrics.get("loss", 0))
            self.metrics_history["perplexity"].append(metrics.get("perplexity", 0))
            self.metrics_history["grad_norm"].append(metrics.get("grad_norm", 0))
            self.metrics_history["bit_stall_rate"].append(
                metrics.get("quantization/bit_stall_rate", 0)
            )
            self.metrics_history["overflow_rate"].append(
                metrics.get("quantization/overflow_rate", 0)
            )

            # Track best loss
            loss = metrics.get("loss", float("inf"))
            if not math.isnan(loss) and loss < self.best_loss:
                self.best_loss = loss

            # Collect diagnostics at interval
            if step % self.config.diagnostic_interval == 0:
                snapshot = self._collect_diagnostics(step, metrics)
                self.diagnostics.append(snapshot)
                self._log_diagnostics_to_wandb(step, snapshot)

            # Check for collapse
            if self._check_collapse(step, metrics):
                # Save checkpoint on collapse
                save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"collapse_step_{step}.pt",
                    step,
                    self.model,
                    self.trainer.optimizer,
                    self.trainer.scaler,
                    self.config,
                )
                break

            # Apply stability interventions
            self._handle_stability_interventions(step, metrics)

            # Regular checkpoint at interval
            if step > 0 and step % self.config.checkpoint_interval == 0:
                self.trainer.checkpoint_manager.save(
                    step,
                    self.model,
                    self.trainer.optimizer,
                    self.trainer.scaler,
                    self.config,
                    loss,
                )

            # Console logging
            if step % self.config.log_interval == 0:
                tps = metrics.get("throughput_tokens_sec", 0)
                stall = metrics.get("quantization/bit_stall_rate", 0)
                print(
                    f"Step {step}: loss={loss:.4f}, grad_norm={metrics.get('grad_norm', 0):.4f}, "
                    f"tok/s={tps:.0f}, stall={stall:.2%}"
                )

            step += 1

        # Calculate duration
        duration = time.time() - self.start_time

        # Build result
        completed = step >= max_steps and self.collapse_step is None
        result = ExperimentResult(
            format_name=self.config.fp8_format,
            completed=completed,
            steps_completed=step,
            max_steps=max_steps,
            final_loss=final_metrics.get("loss", float("inf")),
            best_loss=self.best_loss,
            collapse_step=self.collapse_step,
            collapse_reason=self.collapse_reason,
            format_shifted=self.format_shifted,
            shifted_to=self.shifted_to,
            diagnostics=self.diagnostics,
            duration_seconds=duration,
            metrics_history=self.metrics_history,
        )

        # Generate failure report if collapsed
        if not completed and self.collapse_step is not None:
            report_path = self._generate_failure_report(result, final_metrics)
            result.failure_report_path = report_path

        # Finish W&B run
        if self.trainer.tracker is not None:
            self.trainer.tracker.finish()

        # Final summary
        print("\n" + "=" * 60)
        print(f"Experiment Complete: {self.config.fp8_format}")
        print(f"  Status: {'Completed' if completed else 'Collapsed'}")
        print(f"  Steps: {step}/{max_steps}")
        print(f"  Final loss: {result.final_loss:.4f}")
        print(f"  Best loss: {result.best_loss:.4f}")
        print(f"  Duration: {duration:.1f}s")
        if not completed:
            print(f"  Collapse step: {result.collapse_step}")
            print(f"  Collapse reason: {result.collapse_reason}")
            print(f"  Failure report: {result.failure_report_path}")
        print("=" * 60)

        return result


def run_format_experiment(
    config_path: str,
    device: str = "cuda",
) -> ExperimentResult:
    """Convenience function to run experiment from config file.

    Args:
        config_path: Path to YAML config file
        device: Training device

    Returns:
        ExperimentResult from experiment run
    """
    from altgrad.training.config import load_config
    from altgrad.training.model import GPT, GPTConfig
    from altgrad.training.data import prepare_eurlex

    # Load config
    config = load_config(config_path)

    # Prepare data (idempotent)
    prepare_eurlex("data/eurlex")

    # Create model
    model_config = GPTConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
    )
    model = GPT(model_config)

    # Set seed
    torch.manual_seed(config.seed)
    if "cuda" in device:
        torch.cuda.manual_seed(config.seed)

    # Run experiment
    runner = FormatExperimentRunner(config, model, "data/eurlex", device)
    return runner.run()


__all__ = [
    "FormatExperimentRunner",
    "ExperimentResult",
    "DiagnosticSnapshot",
    "run_format_experiment",
]
