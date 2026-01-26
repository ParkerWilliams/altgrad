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

from typing import Dict

import torch
from torch import Tensor

from altgrad.quantization.formats import FP8Format
from altgrad.quantization.ops import quantize


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
        self.total_weights: Dict[str, int] = {}

    def snapshot_pre_step(
        self,
        name: str,
        weight: Tensor,
        format: FP8Format,
        scale: Tensor,
    ) -> None:
        """Capture quantized weight state before optimizer step.

        Quantizes the weight tensor and stores a clone for later comparison.
        Must be called before optimizer.step() for accurate flip detection.

        Args:
            name: Layer name (unique identifier for tracking)
            weight: Weight tensor to snapshot
            format: FP8 format specification
            scale: Scale factor for quantization

        Example:
            >>> tracker.snapshot_pre_step("layer1", model.layer1.weight, E5M2, scale)
            >>> optimizer.step()
            >>> tracker.compute_flips_post_step("layer1", model.layer1.weight, E5M2, scale)
        """
        # Quantize and store clone (don't hold reference to live tensor)
        q = quantize(weight.detach(), format, scale)
        self.prev_quantized[name] = q.clone()

        # Initialize counters if first time seeing this layer
        if name not in self.flip_counts:
            self.flip_counts[name] = 0
            self.total_weights[name] = weight.numel()

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

    def reset(self) -> None:
        """Clear all tracking state for new epoch or experiment.

        Resets:
          - Pre-step snapshots
          - Flip counts
          - Weight totals

        Example:
            >>> tracker.reset()  # Start fresh for next epoch
            >>> assert tracker.get_flip_counts() == {}
        """
        self.prev_quantized.clear()
        self.flip_counts.clear()
        self.total_weights.clear()


__all__ = [
    "compute_flip_rate",
    "WeightFlipTracker",
]
