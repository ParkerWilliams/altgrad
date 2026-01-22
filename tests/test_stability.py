"""Tests for stability interventions: PartitionRelativeClipper and EmergencyMantissaShift.

STAB-05: Partition-relative gradient clipping - scale clip threshold by format range.
STAB-06: Emergency mantissa shift - fallback to higher-mantissa format on collapse.
"""

import pytest
import torch
import torch.nn as nn

from altgrad.quantization import E3M4, E5M2, E7M0, E1M6, E0M7
from altgrad.quantization.stability import PartitionRelativeClipper, EmergencyMantissaShift


# =============================================================================
# PartitionRelativeClipper Tests (STAB-05)
# =============================================================================


class TestPartitionRelativeClipperThresholdScaling:
    """Test that clip threshold scales by format's dynamic range ratio."""

    def test_clip_threshold_scales_by_format_range(self):
        """E3M4 (max ~124) should have ~0.2% of E5M2 (max 57344) threshold."""
        # E5M2 baseline: max ~57344
        # E3M4: max ~124
        # Ratio: 124 / 57344 = ~0.00216
        clipper_e3m4 = PartitionRelativeClipper(E3M4, base_clip=1.0)
        clipper_e5m2 = PartitionRelativeClipper(E5M2, base_clip=1.0)

        # E3M4 threshold should be approximately 0.2% of E5M2's
        ratio = clipper_e3m4.clip_threshold / clipper_e5m2.clip_threshold
        assert 0.001 < ratio < 0.005, f"Expected ratio ~0.002, got {ratio}"

    def test_clipper_with_e7m0_extreme_range(self):
        """E7M0 has huge max (~2^64), verify threshold scaling."""
        clipper_e7m0 = PartitionRelativeClipper(E7M0, base_clip=1.0)
        clipper_e5m2 = PartitionRelativeClipper(E5M2, base_clip=1.0)

        # E7M0 max is ~2^64 (astronomically large)
        # Its threshold should be much larger than E5M2
        assert clipper_e7m0.clip_threshold > clipper_e5m2.clip_threshold * 1e10

    def test_clipper_preserves_base_clip_for_e5m2(self):
        """E5M2 baseline should have threshold equal to base_clip."""
        clipper = PartitionRelativeClipper(E5M2, base_clip=2.5)
        # E5M2/E5M2 ratio = 1.0, so threshold = base_clip * 1.0
        assert abs(clipper.clip_threshold - 2.5) < 0.01


class TestPartitionRelativeClipperBehavior:
    """Test clipping activation based on overflow threshold."""

    def test_clipper_no_clip_below_overflow_threshold(self):
        """Should not clip when overflow_rate < 0.01."""
        model = nn.Linear(10, 10)
        # Set known gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100  # Large gradients

        clipper = PartitionRelativeClipper(E5M2, base_clip=1.0, overflow_threshold=0.01)

        # Get grad norm before
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        # Reset gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100

        # Should NOT clip because overflow_rate < threshold
        clipped = clipper.clip_if_needed(model, overflow_rate=0.005)

        assert clipped is False

    def test_clipper_clips_above_overflow_threshold(self):
        """Should clip when overflow_rate >= 0.01."""
        model = nn.Linear(10, 10)
        for param in model.parameters():
            param.grad = torch.randn_like(param) * 100  # Large gradients

        clipper = PartitionRelativeClipper(E5M2, base_clip=1.0, overflow_threshold=0.01)

        # Get initial grad norm
        initial_norm = torch.norm(
            torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        ).item()

        # Should clip because overflow_rate >= threshold
        clipped = clipper.clip_if_needed(model, overflow_rate=0.02)

        # Get final grad norm
        final_norm = torch.norm(
            torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        ).item()

        assert clipped is True
        assert final_norm <= clipper.clip_threshold + 0.01  # Allow small tolerance

    def test_clipper_returns_clipped_flag(self):
        """Returns True when clipping applied, False otherwise."""
        model = nn.Linear(10, 10)
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        clipper = PartitionRelativeClipper(E5M2, base_clip=1.0)

        # Below threshold - should return False
        result_no_clip = clipper.clip_if_needed(model, overflow_rate=0.005)
        assert result_no_clip is False

        # Above threshold - should return True
        result_clip = clipper.clip_if_needed(model, overflow_rate=0.02)
        assert result_clip is True


# =============================================================================
# EmergencyMantissaShift Tests (STAB-06)
# =============================================================================


class TestEmergencyMantissaShiftTriggers:
    """Test conditions that trigger format shift."""

    def test_shift_triggers_on_nan_patience(self):
        """Should recommend shift after 3 consecutive NaN batches."""
        shifter = EmergencyMantissaShift(nan_patience=3, stall_threshold=0.5)

        # First two NaNs - no shift yet
        result1 = shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        assert result1 is None
        result2 = shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        assert result2 is None

        # Third NaN - should trigger shift
        result3 = shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        assert result3 == "E5M2"

    def test_shift_triggers_on_high_stall_rate(self):
        """Should recommend shift when stall_rate > 0.5."""
        shifter = EmergencyMantissaShift(nan_patience=3, stall_threshold=0.5)

        # Stall rate exactly at threshold - should NOT trigger (>0.5 required)
        result_at_threshold = shifter.check_and_shift("E1M6", has_nan=False, stall_rate=0.5)
        assert result_at_threshold is None

        # Stall rate above threshold - should trigger
        result_above = shifter.check_and_shift("E1M6", has_nan=False, stall_rate=0.51)
        assert result_above == "E3M4"

    def test_shift_resets_nan_counter_on_valid_batch(self):
        """Non-NaN batch resets consecutive counter."""
        shifter = EmergencyMantissaShift(nan_patience=3)

        # Two NaN batches
        shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        assert shifter.consecutive_nans == 2

        # Valid batch resets counter
        shifter.check_and_shift("E7M0", has_nan=False, stall_rate=0.0)
        assert shifter.consecutive_nans == 0

        # Two more NaNs - still not at patience
        shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        result = shifter.check_and_shift("E7M0", has_nan=False, stall_rate=0.0)  # Reset again
        assert result is None  # No shift because we kept resetting

    def test_shift_returns_none_when_no_shift_needed(self):
        """Returns None if neither trigger met."""
        shifter = EmergencyMantissaShift()

        # Low stall, no NaN - no shift needed
        result = shifter.check_and_shift("E3M4", has_nan=False, stall_rate=0.1)
        assert result is None

    def test_shift_both_triggers_nan_and_stall(self):
        """Either trigger should cause shift."""
        shifter = EmergencyMantissaShift(nan_patience=3, stall_threshold=0.5)

        # High stall should trigger even without NaN patience
        result = shifter.check_and_shift("E1M6", has_nan=False, stall_rate=0.8)
        assert result == "E3M4"


class TestEmergencyMantissaShiftFallbackChain:
    """Test the format fallback chain."""

    def test_fallback_chain_e7m0_to_e5m2(self):
        """E7M0 -> E5M2."""
        shifter = EmergencyMantissaShift(nan_patience=1)

        result = shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        assert result == "E5M2"

    def test_fallback_chain_e1m6_to_e3m4(self):
        """E1M6 -> E3M4."""
        shifter = EmergencyMantissaShift(nan_patience=1)

        result = shifter.check_and_shift("E1M6", has_nan=True, stall_rate=0.0)
        assert result == "E3M4"

    def test_fallback_chain_e0m7_to_e3m4(self):
        """E0M7 -> E3M4."""
        shifter = EmergencyMantissaShift(nan_patience=1)

        result = shifter.check_and_shift("E0M7", has_nan=True, stall_rate=0.0)
        assert result == "E3M4"

    def test_fallback_chain_e5m2_none(self):
        """E5M2 has no fallback (returns None even if triggered)."""
        shifter = EmergencyMantissaShift(nan_patience=1)

        # Even with trigger, E5M2 has no fallback
        result = shifter.check_and_shift("E5M2", has_nan=True, stall_rate=0.0)
        assert result is None

    def test_fallback_chain_e3m4_to_e5m2(self):
        """E3M4 -> E5M2 (one step up in range)."""
        shifter = EmergencyMantissaShift(nan_patience=1)

        result = shifter.check_and_shift("E3M4", has_nan=True, stall_rate=0.0)
        assert result == "E5M2"


class TestEmergencyMantissaShiftReset:
    """Test reset functionality."""

    def test_reset_clears_nan_counter(self):
        """Reset method clears internal state."""
        shifter = EmergencyMantissaShift(nan_patience=3)

        # Accumulate some NaNs
        shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)
        assert shifter.consecutive_nans == 2

        # Reset
        shifter.reset()
        assert shifter.consecutive_nans == 0

    def test_shift_resets_counter_after_successful_shift(self):
        """Counter resets after triggering a shift."""
        shifter = EmergencyMantissaShift(nan_patience=3)

        # Trigger shift
        for _ in range(3):
            shifter.check_and_shift("E7M0", has_nan=True, stall_rate=0.0)

        # Counter should reset after shift
        assert shifter.consecutive_nans == 0
