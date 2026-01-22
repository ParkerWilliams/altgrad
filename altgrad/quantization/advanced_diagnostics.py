"""Advanced quantization diagnostics for format analysis.

DIAG-01: Stiffness field - minimum meaningful update size at each weight magnitude
DIAG-02: Grid alignment - distance from weights to nearest quantization levels
DIAG-03: Gradient-stiffness correlation - alignment between gradients and precision
DIAG-04: ULP statistics - bit-position movement per update

These diagnostics enable scientific analysis of format suitability.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from altgrad.quantization.formats import FP8Format
from altgrad.quantization.ops import quantize


def compute_stiffness_field(weights: Tensor, mantissa_bits: int) -> Tensor:
    """Compute per-weight stiffness factor (DIAG-01).

    Stiffness S = 2^(floor(log2|w|) - M) represents the minimum
    meaningful update size at each weight's magnitude. Updates smaller
    than S will round to zero (bit-stall).

    Special case: E0M7 (mantissa_bits=7, fixed-point) has constant
    stiffness of 1/128 since the grid is uniformly spaced.

    Args:
        weights: Weight tensor
        mantissa_bits: Number of mantissa bits in format (M)

    Returns:
        Tensor of stiffness values, same shape as weights.
        Zero weights return NaN (undefined stiffness).

    Example:
        >>> w = torch.tensor([1.0, 0.5, 0.25])
        >>> S = compute_stiffness_field(w, mantissa_bits=2)  # E5M2
        >>> # S[0] = 2^(0-2) = 0.25, S[1] = 2^(-1-2) = 0.125, ...
    """
    # E0M7 special case: fixed-point with uniform grid spacing
    if mantissa_bits == 7:
        # Constant stiffness = 1/128 for all non-zero weights
        stiffness = torch.full_like(weights, 1.0 / 128.0)
        stiffness = torch.where(
            weights == 0,
            torch.tensor(float('nan'), dtype=weights.dtype, device=weights.device),
            stiffness
        )
        return stiffness

    # Standard floating-point: S = 2^(floor(log2|w|) - M)
    abs_w = weights.abs()
    # Handle zero weights: set to small positive to avoid log2(0)
    safe_abs_w = abs_w.clamp(min=1e-45)
    log2_w = torch.floor(torch.log2(safe_abs_w))

    stiffness = torch.pow(2.0, log2_w - mantissa_bits)

    # Zero weights have undefined stiffness
    stiffness = torch.where(
        weights == 0,
        torch.tensor(float('nan'), dtype=weights.dtype, device=weights.device),
        stiffness
    )

    return stiffness


def grid_alignment_error(
    weights: Tensor,
    format: FP8Format,
    scale: Tensor
) -> Tensor:
    """Compute distance to nearest quantization grid point (DIAG-02).

    Measures how far each weight is from its nearest representable value.
    Weights exactly on the grid have zero error.

    Args:
        weights: Weight tensor
        format: FP8 format specification
        scale: Scale factor for quantization

    Returns:
        Tensor of absolute errors, same shape as weights.

    Example:
        >>> from altgrad.quantization import E5M2
        >>> error = grid_alignment_error(weights, E5M2, torch.tensor(1.0))
    """
    quantized = quantize(weights, format, scale)
    return torch.abs(weights - quantized)


def grid_alignment_statistics(
    weights: Tensor,
    format: FP8Format,
    scale: Tensor
) -> Dict[str, float]:
    """Return summary statistics for grid alignment (DIAG-02).

    Args:
        weights: Weight tensor
        format: FP8 format specification
        scale: Scale factor for quantization

    Returns:
        Dictionary with keys:
          - grid_error_mean: Mean alignment error
          - grid_error_max: Maximum alignment error
          - grid_error_std: Standard deviation of error
          - on_grid_frac: Fraction of weights exactly on grid
    """
    error = grid_alignment_error(weights, format, scale)
    return {
        "grid_error_mean": error.mean().item(),
        "grid_error_max": error.max().item(),
        "grid_error_std": error.std().item(),
        "on_grid_frac": (error < 1e-10).float().mean().item(),
    }


def compute_ulp_distance(before: Tensor, after: Tensor) -> Tensor:
    """Compute how many ULPs each weight moved (DIAG-04).

    Stub implementation - will be completed in Task 3.
    """
    raise NotImplementedError("compute_ulp_distance not yet implemented")


def ulp_statistics(before: Tensor, after: Tensor) -> Dict[str, float]:
    """Return ULP movement statistics (DIAG-04).

    Stub implementation - will be completed in Task 3.
    """
    raise NotImplementedError("ulp_statistics not yet implemented")


def gradient_stiffness_correlation(
    weights: Tensor,
    gradients: Tensor,
    mantissa_bits: int
) -> Dict[str, float]:
    """Analyze correlation between gradient magnitude and stiffness (DIAG-03).

    Stub implementation - will be completed in Task 3.
    """
    raise NotImplementedError("gradient_stiffness_correlation not yet implemented")


__all__ = [
    "compute_stiffness_field",
    "grid_alignment_error",
    "grid_alignment_statistics",
    "compute_ulp_distance",
    "ulp_statistics",
    "gradient_stiffness_correlation",
]
