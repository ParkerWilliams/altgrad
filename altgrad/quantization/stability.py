"""Stability interventions for exotic FP8 format training.

STAB-05: Partition-relative gradient clipping - scale clip threshold by format's
dynamic range relative to E5M2 baseline.

STAB-06: Emergency mantissa shift - fallback to higher-mantissa format when
training shows persistent NaN or high bit-stall rate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from altgrad.quantization.formats import FP8Format, FORMAT_REGISTRY

# E5M2 max representable value (baseline for scaling)
# E5M2 has max_exp=30 (31-1 for inf), mantissa=3/4, so 2^15 * 1.75 = 57344
E5M2_MAX = 57344.0


class PartitionRelativeClipper:
    """Format-aware gradient clipping (STAB-05).

    Scales gradient clip threshold proportionally to format's dynamic range.
    Formats with smaller range (E3M4, E1M6) get proportionally smaller thresholds.

    Only clips when overflow rate exceeds threshold (not always-on).

    Example:
        >>> clipper = PartitionRelativeClipper(E3M4, base_clip=1.0)
        >>> clipped = clipper.clip_if_needed(model, overflow_rate=0.02)
    """

    def __init__(
        self,
        format: FP8Format,
        base_clip: float = 1.0,
        overflow_threshold: float = 0.01,  # 1% from CONTEXT.md
    ):
        """Initialize clipper with format-scaled threshold.

        Args:
            format: FP8 format specification
            base_clip: Base clipping threshold (for E5M2 baseline)
            overflow_threshold: Minimum overflow rate to activate clipping [0, 1]
        """
        self.format = format
        self.base_clip = base_clip
        self.overflow_threshold = overflow_threshold

        # Scale threshold by format's range ratio vs E5M2
        format_max = format.max_representable_value
        self.clip_threshold = base_clip * (format_max / E5M2_MAX)

    def clip_if_needed(self, model: nn.Module, overflow_rate: float) -> bool:
        """Apply clipping only when overflow detected.

        Args:
            model: Model to clip gradients on
            overflow_rate: Fraction of values that overflowed [0, 1]

        Returns:
            True if clipping was applied, False otherwise
        """
        if overflow_rate >= self.overflow_threshold:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_threshold)
            return True
        return False


class EmergencyMantissaShift:
    """Emergency fallback to higher-mantissa format (STAB-06).

    Monitors for:
    - Persistent NaN (3+ consecutive batches with NaN loss)
    - High bit-stall rate (>50% of updates stalled)

    When triggered, recommends fallback to format with more mantissa bits:
    - E7M0 -> E5M2 (powers-of-2 -> standard FP8)
    - E1M6 -> E3M4 (narrow range -> moderate)
    - E0M7 -> E3M4 (fixed-point -> floating)
    - E3M4 -> E5M2 (fallback for E3M4 if needed)
    - E5M2 -> None (no fallback, already widest range)

    Example:
        >>> shifter = EmergencyMantissaShift()
        >>> new_format = shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.3)
        >>> # Returns "E5M2" after 3 consecutive NaN batches
    """

    # Fallback chain: format -> fallback format with more mantissa/range
    FORMAT_FALLBACK = {
        "E7M0": "E5M2",  # 0 mantissa -> 2 mantissa
        "E1M6": "E3M4",  # 6 mantissa but tiny range -> 4 mantissa moderate range
        "E0M7": "E3M4",  # Fixed-point -> floating
        "E3M4": "E5M2",  # Moderate -> standard
        "E5M2": None,  # No fallback (widest stable format)
    }

    def __init__(
        self,
        nan_patience: int = 3,
        stall_threshold: float = 0.5,
    ):
        """Initialize shift monitor.

        Args:
            nan_patience: Number of consecutive NaN batches before triggering shift
            stall_threshold: Bit-stall rate above which to trigger shift (>threshold)
        """
        self.nan_patience = nan_patience
        self.stall_threshold = stall_threshold
        self.consecutive_nans = 0

    def check_and_shift(
        self,
        current_format: str,
        has_nan: bool,
        stall_rate: float,
    ) -> Optional[str]:
        """Check if format shift is needed and return new format.

        Args:
            current_format: Current format name (e.g., "E7M0")
            has_nan: Whether current batch had NaN loss
            stall_rate: Fraction of bit-stalled updates [0, 1]

        Returns:
            New format name if shift needed, None otherwise
        """
        # Track consecutive NaNs
        if has_nan:
            self.consecutive_nans += 1
        else:
            self.consecutive_nans = 0

        # Check triggers
        nan_triggered = self.consecutive_nans >= self.nan_patience
        stall_triggered = stall_rate > self.stall_threshold

        should_shift = nan_triggered or stall_triggered

        if should_shift:
            fallback = self.FORMAT_FALLBACK.get(current_format)
            if fallback is not None:
                self.consecutive_nans = 0  # Reset counter after shift
            return fallback

        return None

    def reset(self) -> None:
        """Reset internal state (call after format shift or run start)."""
        self.consecutive_nans = 0


__all__ = [
    "PartitionRelativeClipper",
    "EmergencyMantissaShift",
]
