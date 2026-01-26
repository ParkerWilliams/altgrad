"""Weight flip metrics for tracking quantization state transitions.

This module provides tools for counting how often quantized weights change
their FP8 representation during training. Flip counting complements bit-stall
detection: while stalls identify when updates are too small to change, flips
track actual transitions.

Why Weight Flips Matter:
  High flip rate = weights actively evolving in quantized space
  Low flip rate = weights stuck at same quantized values (possible learning plateau)
  Zero flips = complete training stall

Relationship to Bit-Stall:
  - Bit-stall: Would this update change the quantized value?
  - Weight flip: Did the quantized value actually change?

  Bit-stall is predictive (before step), flip is observational (after step).

Example:
    >>> tracker = WeightFlipTracker()
    >>> for step in range(100):
    ...     tracker.snapshot_pre_step("layer1", model.layer1.weight, E5M2, scale)
    ...     optimizer.step()
    ...     flips = tracker.compute_flips_post_step("layer1", model.layer1.weight, E5M2, scale)
    ...     print(f"Step {step}: {flips} flips")
    >>> print(f"Total flip rate: {tracker.get_flip_rates()}")
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from altgrad.quantization.formats import FP8Format
from altgrad.quantization.ops import quantize


def compute_stall_ratio(flips: int, updates: int) -> float:
    """Compute stall ratio from flip and update counts.

    Stall ratio measures what fraction of attempted updates failed to change
    the quantized representation. High stall ratio indicates gradient steps
    too small to overcome quantization granularity.

    Args:
        flips: Number of weights that changed FP8 representation
        updates: Number of weights that received non-zero gradients

    Returns:
        Stall ratio in [0.0, 1.0] where:
          - 0.0 = all updates caused flips (ideal)
          - 1.0 = no updates caused flips (complete stall)
          - 0.0 if updates == 0 (no gradient = no stall, not error)

    Example:
        >>> compute_stall_ratio(10, 100)  # 10 flips from 100 updates
        0.9  # 90% stall rate
        >>> compute_stall_ratio(0, 0)  # No gradient
        0.0  # No stall (by definition)
    """
    if updates == 0:
        return 0.0
    return 1.0 - (flips / updates)


def compute_flip_rate(prev_quantized: Tensor, curr_quantized: Tensor) -> float:
    """Compute flip rate between two quantized tensors.

    A flip occurs when the quantized representation changes between tensors.
    This is a standalone utility for direct comparison of quantized states.

    Args:
        prev_quantized: Previously quantized tensor
        curr_quantized: Current quantized tensor (must be same shape)

    Returns:
        Flip rate in [0.0, 1.0] where:
          - 0.0 = no elements changed
          - 1.0 = all elements changed

    Raises:
        ValueError: If tensors have different shapes

    Example:
        >>> q1 = torch.tensor([1.0, 2.0, 3.0])
        >>> q2 = torch.tensor([1.0, 5.0, 6.0])  # 2 of 3 changed
        >>> compute_flip_rate(q1, q2)
        0.6666...
    """
    if prev_quantized.shape != curr_quantized.shape:
        raise ValueError(
            f"Shape mismatch: prev={prev_quantized.shape}, curr={curr_quantized.shape}"
        )

    total = prev_quantized.numel()
    if total == 0:
        return 0.0

    flips = (prev_quantized != curr_quantized).sum().item()
    return flips / total


class WeightFlipTracker:
    """Tracks weight transitions between FP8 representations across training.

    Captures quantized weight state before optimizer step, then counts how many
    weights changed their FP8 representation after the step. Provides per-layer
    flip counts and rates for training diagnostics.

    Attributes:
        prev_quantized: Dict mapping layer names to their pre-step quantized state
        flip_counts: Dict mapping layer names to cumulative flip counts
        update_counts: Dict mapping layer names to cumulative update counts
        total_weights: Dict mapping layer names to their weight count

    Example:
        >>> tracker = WeightFlipTracker()
        >>> # Before optimizer step
        >>> tracker.snapshot_pre_step("encoder.layer1", weight, E5M2, scale)
        >>> # After optimizer step
        >>> flips = tracker.compute_flips_post_step("encoder.layer1", weight, E5M2, scale)
        >>> print(f"Flips: {flips}, Rate: {tracker.get_flip_rates()['encoder.layer1']:.2%}")
    """

    def __init__(self) -> None:
        """Initialize empty tracking state."""
        self.prev_quantized: Dict[str, Tensor] = {}
        self.flip_counts: Dict[str, int] = {}
        self.update_counts: Dict[str, int] = {}
        self.total_weights: Dict[str, int] = {}

    def snapshot_pre_step(
        self,
        name: str,
        weight: Tensor,
        format: FP8Format,
        scale: Tensor,
        grad: Optional[Tensor] = None,
    ) -> None:
        """Capture quantized weight state before optimizer step.

        Quantizes the weight tensor and stores a clone for later comparison.
        Must be called before optimizer.step() for accurate flip detection.

        Optionally tracks update counts if gradient tensor is provided. Updates
        are counted as weights with non-zero gradients (|grad| > 1e-10).

        Args:
            name: Layer name (unique identifier for tracking)
            weight: Weight tensor to snapshot
            format: FP8 format specification
            scale: Scale factor for quantization
            grad: Optional gradient tensor for update counting

        Example:
            >>> tracker.snapshot_pre_step("layer1", model.layer1.weight, E5M2, scale, weight.grad)
            >>> optimizer.step()
            >>> tracker.compute_flips_post_step("layer1", model.layer1.weight, E5M2, scale)
        """
        # Quantize and store clone (don't hold reference to live tensor)
        q = quantize(weight.detach(), format, scale)
        self.prev_quantized[name] = q.clone()

        # Initialize counters if first time seeing this layer
        if name not in self.flip_counts:
            self.flip_counts[name] = 0
            self.update_counts[name] = 0
            self.total_weights[name] = weight.numel()

        # Count updates if gradient provided
        if grad is not None:
            updates = (grad.abs() > 1e-10).sum().item()
            self.update_counts[name] += int(updates)

    def compute_flips_post_step(
        self,
        name: str,
        weight: Tensor,
        format: FP8Format,
        scale: Tensor,
    ) -> int:
        """Count weights that changed FP8 representation after optimizer step.

        Compares current quantized state to the snapshot taken before step.
        Adds flip count to cumulative total for this layer.

        Args:
            name: Layer name (must match snapshot_pre_step)
            weight: Weight tensor after optimizer step
            format: FP8 format specification (same as snapshot)
            scale: Scale factor for quantization (same as snapshot)

        Returns:
            Number of weights that changed quantized representation

        Raises:
            KeyError: If snapshot_pre_step was not called for this layer

        Example:
            >>> tracker.snapshot_pre_step("layer1", w_before, E5M2, scale)
            >>> optimizer.step()
            >>> flips = tracker.compute_flips_post_step("layer1", w_after, E5M2, scale)
        """
        if name not in self.prev_quantized:
            raise KeyError(
                f"No pre-step snapshot for '{name}'. "
                "Call snapshot_pre_step before optimizer.step()."
            )

        # Quantize current state
        q_curr = quantize(weight.detach(), format, scale)
        q_prev = self.prev_quantized[name]

        # Count flips (element-wise comparison)
        flips = (q_prev != q_curr).sum().item()

        # Accumulate
        self.flip_counts[name] += int(flips)

        # Clean up snapshot (optional: could keep for multi-step analysis)
        del self.prev_quantized[name]

        return int(flips)

    def get_flip_counts(self) -> Dict[str, int]:
        """Get cumulative per-layer flip counts.

        Returns:
            Dictionary mapping layer names to total flip counts since last reset

        Example:
            >>> counts = tracker.get_flip_counts()
            >>> for name, count in counts.items():
            ...     print(f"{name}: {count} flips")
        """
        return dict(self.flip_counts)

    def get_flip_rates(self) -> Dict[str, float]:
        """Get per-layer flip rates (flips / total_weights).

        Returns:
            Dictionary mapping layer names to flip rates in [0.0, 1.0]
            Note: Rate can exceed 1.0 over multiple steps (cumulative)

        Example:
            >>> rates = tracker.get_flip_rates()
            >>> for name, rate in rates.items():
            ...     print(f"{name}: {rate:.2%} flip rate")
        """
        rates = {}
        for name, count in self.flip_counts.items():
            total = self.total_weights.get(name, 0)
            if total > 0:
                rates[name] = count / total
            else:
                rates[name] = 0.0
        return rates

    def get_update_counts(self) -> Dict[str, int]:
        """Get cumulative per-layer update counts.

        Update count tracks how many weights received non-zero gradients.
        This represents attempted updates, regardless of whether they
        caused FP8 representation changes.

        Returns:
            Dictionary mapping layer names to total update counts since last reset

        Example:
            >>> counts = tracker.get_update_counts()
            >>> for name, count in counts.items():
            ...     print(f"{name}: {count} updates attempted")
        """
        return dict(self.update_counts)

    def get_stall_ratios(self) -> Dict[str, float]:
        """Get per-layer stall ratios (1 - flips/updates).

        Stall ratio measures what fraction of gradient updates failed to
        change the FP8 representation. High stall indicates gradient steps
        too small to overcome quantization granularity.

        Returns:
            Dictionary mapping layer names to stall ratios in [0.0, 1.0]:
              - 0.0 = all updates caused flips (ideal)
              - 1.0 = no updates caused flips (complete stall)
              - 0.0 if no updates (no gradient = no stall)

        Example:
            >>> ratios = tracker.get_stall_ratios()
            >>> for name, ratio in ratios.items():
            ...     print(f"{name}: {ratio:.1%} stall rate")
        """
        ratios = {}
        for name in self.flip_counts:
            flips = self.flip_counts.get(name, 0)
            updates = self.update_counts.get(name, 0)
            ratios[name] = compute_stall_ratio(flips, updates)
        return ratios

    def reset(self) -> None:
        """Clear all tracking state for new epoch or experiment.

        Resets:
          - Pre-step snapshots
          - Flip counts
          - Update counts
          - Weight totals

        Example:
            >>> tracker.reset()  # Start fresh for next epoch
            >>> assert tracker.get_flip_counts() == {}
        """
        self.prev_quantized.clear()
        self.flip_counts.clear()
        self.update_counts.clear()
        self.total_weights.clear()


__all__ = [
    "compute_flip_rate",
    "compute_stall_ratio",
    "WeightFlipTracker",
]
