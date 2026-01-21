"""Per-tensor dynamic scaling with amax history tracking.

This module provides dynamic scaling to map tensor ranges to FP8 representable ranges.
The amax history buffer smooths scale updates across batches to reduce noise.

Scaling Approach:
  scale = amax(tensor) / format.max_representable_value

Where amax is tracked over a sliding window of recent batches.
This delayed scaling approach (from NVIDIA FP8 training) balances stability vs. adaptation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from altgrad.quantization.formats import FP8Format


@dataclass
class ScalingConfig:
    """Configuration for dynamic scaling behavior.

    Attributes:
        history_len: Number of recent amax values to track (default: 16)
        scale_min: Minimum scale value to prevent division by zero (default: 1e-10)
    """

    history_len: int = 16
    scale_min: float = 1e-10


class AmaxHistory:
    """Tracks maximum absolute values over recent batches.

    Maintains a sliding window of amax values and computes scale factors
    for FP8 quantization. This smooths scale updates and reduces noise
    from batch-to-batch variation.

    Example:
        >>> history = AmaxHistory(history_len=16)
        >>> for batch in batches:
        ...     history.update(batch)
        ...     scale = compute_scale(history.get_amax(), E5M2)
        ...     quantized = quantize(batch, E5M2, torch.tensor(scale))
    """

    def __init__(self, history_len: int = 16):
        """Initialize amax history buffer.

        Args:
            history_len: Maximum number of amax values to track
        """
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        self.history: deque = deque(maxlen=history_len)

    def update(self, tensor: Tensor) -> None:
        """Record maximum absolute value from tensor.

        Args:
            tensor: Input tensor to compute amax from
        """
        amax_val = tensor.abs().amax().item()
        self.history.append(amax_val)

    def get_amax(self) -> float:
        """Get maximum amax from history.

        Returns:
            Maximum of all tracked amax values, or 1.0 if history is empty
        """
        if not self.history:
            return 1.0
        return max(self.history)

    def reset(self) -> None:
        """Clear all history."""
        self.history.clear()

    def __len__(self) -> int:
        """Return number of amax values currently tracked."""
        return len(self.history)


def compute_scale(amax: float, format: FP8Format, scale_min: float = 1e-10) -> float:
    """Compute scale factor for quantization.

    The scale maps the tensor's dynamic range to the FP8 representable range:
      scale = amax / format.max_representable_value

    Args:
        amax: Maximum absolute value of tensor
        format: FP8 format specification
        scale_min: Minimum scale to prevent division by zero

    Returns:
        Scale factor for quantization

    Example:
        >>> scale = compute_scale(10.0, E5M2)
        >>> # Maps tensor in [-10, 10] to E5M2 range
    """
    scale = amax / format.max_representable_value
    return max(scale, scale_min)


__all__ = [
    "ScalingConfig",
    "AmaxHistory",
    "compute_scale",
]
