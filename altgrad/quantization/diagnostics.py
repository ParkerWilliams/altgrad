"""Bit-stall detection and quantization diagnostics.

This module provides diagnostic tools for understanding quantization behavior.
Bit-stall detection identifies when gradient updates are too small to change
quantized weight values -- a critical signal for format suitability.

Bit-Stall Phenomenon:
  When round(w_quantized + update_quantized) == w_quantized despite non-zero gradient
  This occurs when the update magnitude is smaller than the quantization ULP (unit in last place)

Why it matters:
  High stall rate indicates:
  - Format lacks precision for this magnitude range
  - Learning is effectively frozen (weights don't change)
  - Need different format or adaptive precision

Detection approach:
  1. Quantize current weight: w_q = quantize(w, format, scale)
  2. Quantize updated weight: w_new_q = quantize(w - lr * grad, format, scale)
  3. Stall if w_q == w_new_q but grad != 0
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from altgrad.quantization.formats import FP8Format
from altgrad.quantization.ops import quantize


def detect_bit_stall(
    weight: Tensor,
    grad: Tensor,
    lr: float,
    format: FP8Format,
    scale: Tensor,
) -> tuple[int, int]:
    """Detect bit-stall: quantized updates that round to zero.

    A bit-stall occurs when the quantized update doesn't change the quantized weight
    despite having a non-zero gradient. This indicates the gradient magnitude is
    smaller than the quantization precision at that weight value.

    Args:
        weight: Current weight tensor
        grad: Gradient tensor (same shape as weight)
        lr: Learning rate
        format: FP8 format specification
        scale: Scale factor for quantization

    Returns:
        Tuple of (stall_count, total_count) where:
          - stall_count: Number of weights that would stall
          - total_count: Total number of weights with non-zero gradients

    Example:
        >>> w = torch.randn(100)
        >>> g = torch.randn(100) * 1e-5  # Very small gradients
        >>> stall_count, total = detect_bit_stall(w, g, 0.01, E5M2, torch.tensor(1.0))
        >>> stall_rate = stall_count / total
    """
    # Compute proposed update: SGD step
    update = -lr * grad

    # Quantize current weight
    w_q = quantize(weight, format, scale)

    # Quantize updated weight
    w_new_q = quantize(weight + update, format, scale)

    # Identify stalls: quantized values didn't change despite non-zero gradient
    # Use small threshold for "non-zero" to handle floating point precision
    has_gradient = grad.abs() > 1e-10
    no_change = (w_q == w_new_q)
    stall_mask = has_gradient & no_change

    stall_count = stall_mask.sum().item()
    total_count = has_gradient.sum().item()

    return stall_count, total_count


class BitStallDetector:
    """Accumulates bit-stall statistics over multiple updates.

    Tracks how often quantized weight updates round to zero across
    training steps. High stall rates indicate format precision issues.

    Example:
        >>> detector = BitStallDetector()
        >>> for batch in batches:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     for param in model.parameters():
        ...         detector.update(param.data, param.grad, lr, E5M2, scale)
        ...     optimizer.step()
        >>> print(f"Stall rate: {detector.get_stall_rate():.2%}")
    """

    def __init__(self):
        """Initialize bit-stall counters."""
        self.stall_count: int = 0
        self.total_count: int = 0
        self.step_count: int = 0

    def update(
        self,
        weight: Tensor,
        grad: Tensor,
        lr: float,
        format: FP8Format,
        scale: Tensor,
    ) -> None:
        """Record stall statistics for one update.

        Args:
            weight: Current weight tensor
            grad: Gradient tensor
            lr: Learning rate
            format: FP8 format specification
            scale: Scale factor for quantization
        """
        stall, total = detect_bit_stall(weight, grad, lr, format, scale)
        self.stall_count += stall
        self.total_count += total
        self.step_count += 1

    def get_stall_rate(self) -> float:
        """Get fraction of updates that stalled.

        Returns:
            Stall rate in [0, 1], or 0.0 if no updates recorded
        """
        if self.total_count == 0:
            return 0.0
        return self.stall_count / self.total_count

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.stall_count = 0
        self.total_count = 0
        self.step_count = 0

    def get_stats(self) -> Dict[str, float]:
        """Get detailed statistics dictionary.

        Returns:
            Dictionary with keys:
              - stall_rate: Fraction of stalled updates [0, 1]
              - stall_count: Total number of stalled updates
              - total_count: Total number of updates with gradients
              - steps: Number of update steps recorded
        """
        return {
            "stall_rate": self.get_stall_rate(),
            "stall_count": self.stall_count,
            "total_count": self.total_count,
            "steps": self.step_count,
        }


__all__ = [
    "detect_bit_stall",
    "BitStallDetector",
]
