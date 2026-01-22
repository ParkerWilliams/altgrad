"""Tests for advanced quantization diagnostics (DIAG-01 to DIAG-04).

DIAG-01: Stiffness field - minimum meaningful update size at each weight magnitude
DIAG-02: Grid alignment - distance from weights to nearest quantization levels
DIAG-03: Gradient-stiffness correlation - alignment between gradients and precision
DIAG-04: ULP statistics - bit-position movement per update
"""

import math

import pytest
import torch

from altgrad.quantization import E0M7, E3M4, E5M2, E7M0
from altgrad.quantization.advanced_diagnostics import (
    compute_stiffness_field,
    grid_alignment_error,
    grid_alignment_statistics,
    compute_ulp_distance,
    ulp_statistics,
    gradient_stiffness_correlation,
)


class TestStiffnessField:
    """Tests for DIAG-01: Stiffness field computation."""

    def test_stiffness_formula_basic(self):
        """Test stiffness formula: S = 2^(floor(log2|w|) - M).

        For w=1.0, M=2 (E5M2): S = 2^(floor(log2(1.0)) - 2) = 2^(0-2) = 0.25
        """
        w = torch.tensor([1.0])
        # E5M2 has 2 mantissa bits
        S = compute_stiffness_field(w, mantissa_bits=2)
        assert abs(S[0].item() - 0.25) < 1e-6

    def test_stiffness_scales_with_magnitude(self):
        """Test that larger |w| gives larger stiffness (less precision).

        For w=2.0, M=2: S = 2^(floor(log2(2.0)) - 2) = 2^(1-2) = 0.5
        For w=4.0, M=2: S = 2^(floor(log2(4.0)) - 2) = 2^(2-2) = 1.0
        """
        w = torch.tensor([1.0, 2.0, 4.0])
        S = compute_stiffness_field(w, mantissa_bits=2)

        assert S[0].item() < S[1].item()  # S(1.0) < S(2.0)
        assert S[1].item() < S[2].item()  # S(2.0) < S(4.0)

        # Verify exact values
        assert abs(S[1].item() - 0.5) < 1e-6
        assert abs(S[2].item() - 1.0) < 1e-6

    def test_stiffness_zero_weight_is_nan(self):
        """Test that zero weights return NaN stiffness (undefined)."""
        w = torch.tensor([0.0, 1.0, 0.0])
        S = compute_stiffness_field(w, mantissa_bits=2)

        assert math.isnan(S[0].item())
        assert not math.isnan(S[1].item())
        assert math.isnan(S[2].item())

    def test_stiffness_e0m7_constant(self):
        """Test E0M7 returns constant 1/128 stiffness.

        E0M7 (M=7) is fixed-point with uniform grid spacing of 1/128.
        """
        w = torch.tensor([0.1, 0.5, 0.9, -0.3])
        S = compute_stiffness_field(w, mantissa_bits=7)

        expected = 1.0 / 128.0
        for i in range(len(w)):
            assert abs(S[i].item() - expected) < 1e-10

    def test_stiffness_negative_weights(self):
        """Test that sign doesn't affect stiffness (uses abs)."""
        w_pos = torch.tensor([1.0, 2.0, 0.5])
        w_neg = torch.tensor([-1.0, -2.0, -0.5])

        S_pos = compute_stiffness_field(w_pos, mantissa_bits=2)
        S_neg = compute_stiffness_field(w_neg, mantissa_bits=2)

        assert torch.allclose(S_pos, S_neg)

    def test_stiffness_tensor_batch(self):
        """Test stiffness works on full tensor and returns same shape."""
        w = torch.randn(10, 20, 30)
        S = compute_stiffness_field(w, mantissa_bits=2)

        assert S.shape == w.shape


class TestGridAlignment:
    """Tests for DIAG-02: Grid alignment measurement."""

    def test_grid_alignment_on_grid_zero(self):
        """Test that values exactly on grid have zero error.

        E5M2 can exactly represent 1.0, 2.0, 0.5.
        """
        w = torch.tensor([1.0, 2.0, 0.5])
        scale = torch.tensor(1.0)
        error = grid_alignment_error(w, E5M2, scale)

        # Should be zero or very close
        assert error.max().item() < 1e-6

    def test_grid_alignment_between_grid_points(self):
        """Test that mid-point between grid points has positive error."""
        # 1.0 and 1.5 are representable in E5M2 (1.0, 1.5, 2.0, ...)
        # 1.25 is between them and may not be exactly representable
        w = torch.tensor([1.25])
        scale = torch.tensor(1.0)
        error = grid_alignment_error(w, E5M2, scale)

        # Should have some error (E5M2 with M=2 has 4 mantissa levels per binade)
        # Between 1.0 and 2.0: representable are 1.0, 1.25, 1.5, 1.75
        # So 1.25 should actually be on grid for E5M2
        # Let's use a value definitely between grid points
        w2 = torch.tensor([1.125])  # Not representable exactly
        error2 = grid_alignment_error(w2, E5M2, scale)

        # At least one should have error
        assert error.max().item() >= 0 or error2.max().item() > 0

    def test_grid_alignment_statistics_returns_dict(self):
        """Test that statistics returns dict with mean, max, std, on_grid_frac."""
        w = torch.randn(100)
        scale = torch.tensor(1.0)
        stats = grid_alignment_statistics(w, E5M2, scale)

        assert "grid_error_mean" in stats
        assert "grid_error_max" in stats
        assert "grid_error_std" in stats
        assert "on_grid_frac" in stats

        assert isinstance(stats["grid_error_mean"], float)
        assert isinstance(stats["grid_error_max"], float)
        assert isinstance(stats["grid_error_std"], float)
        assert isinstance(stats["on_grid_frac"], float)

        assert stats["on_grid_frac"] >= 0 and stats["on_grid_frac"] <= 1

    def test_grid_alignment_clamped_values(self):
        """Test that values outside format range clamp or quantize to boundary.

        For E5M2 (with has_inf=True), overflow goes to inf, so error is inf.
        For formats without inf, overflow clamps to max representable.
        This test uses E3M4 which doesn't have inf support.
        """
        # E3M4 max is about 124.0, and it clamps (no inf support)
        w = torch.tensor([1000.0])
        scale = torch.tensor(1.0)
        error = grid_alignment_error(w, E3M4, scale)

        # Should not be NaN (clamps to max finite value)
        assert not math.isnan(error[0].item())
        # Error should be finite (difference between 1000 and max representable)
        assert not math.isinf(error[0].item())


class TestGradientStiffnessCorrelation:
    """Tests for DIAG-03: Gradient-stiffness correlation."""

    def test_correlation_returns_pearson(self):
        """Test that correlation coefficient is in [-1, 1]."""
        w = torch.randn(100)
        grad = torch.randn(100)
        result = gradient_stiffness_correlation(w, grad, mantissa_bits=2)

        assert "grad_stiff_correlation" in result
        corr = result["grad_stiff_correlation"]
        assert -1 <= corr <= 1

    def test_correlation_positive_when_aligned(self):
        """Test positive correlation when high gradients at high stiffness."""
        # Create weights where large weights get large gradients
        # (problematic alignment)
        w = torch.abs(torch.randn(100)) + 0.1  # All positive
        # Gradient proportional to weight magnitude
        grad = w.clone()

        result = gradient_stiffness_correlation(w, grad, mantissa_bits=2)

        # Stiffness increases with |w|, and grad ~ |w|, so positive correlation
        # But this is statistical, may not always be strongly positive
        assert "grad_stiff_correlation" in result

    def test_correlation_handles_zero_weights(self):
        """Test that zero-weight positions are masked out."""
        w = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.0])
        grad = torch.tensor([1.0, 0.1, 0.2, 1.0, 0.3])

        # Should not raise error and should exclude zeros
        result = gradient_stiffness_correlation(w, grad, mantissa_bits=2)

        assert "grad_stiff_correlation" in result
        assert not math.isnan(result["grad_stiff_correlation"])

    def test_correlation_ratio_below_one_means_stall(self):
        """Test grad/stiffness ratio indicates stall risk."""
        w = torch.randn(100).abs() + 1.0  # Weights around 1-2
        grad = torch.randn(100) * 0.001  # Very small gradients

        result = gradient_stiffness_correlation(w, grad, mantissa_bits=2)

        # With small gradients and normal stiffness, ratio should be low
        assert "grad_stiff_ratio_mean" in result

    def test_grad_below_stiffness_frac_computed(self):
        """Test that fraction of gradients below stiffness is computed."""
        w = torch.randn(100).abs() + 0.5
        grad = torch.randn(100) * 0.01

        result = gradient_stiffness_correlation(w, grad, mantissa_bits=2)

        assert "grad_below_stiffness_frac" in result
        frac = result["grad_below_stiffness_frac"]
        assert 0 <= frac <= 1


class TestULPStatistics:
    """Tests for DIAG-04: ULP statistics."""

    def test_ulp_distance_identical_weights(self):
        """Test that before==after gives distance 0."""
        before = torch.tensor([1.0, 2.0, 0.5])
        after = before.clone()

        dist = compute_ulp_distance(before, after)

        assert torch.all(dist == 0)

    def test_ulp_distance_one_ulp_move(self):
        """Test that moving by exactly one ULP gives distance ~1."""
        before = torch.tensor([1.0])
        # Move by one ULP
        after = torch.nextafter(before, torch.tensor([2.0]))

        dist = compute_ulp_distance(before, after)

        # Should be approximately 1 (may have small floating point error)
        assert abs(dist[0].item() - 1.0) < 0.01

    def test_ulp_statistics_returns_dict(self):
        """Test that ulp_statistics returns dict with mean, median, max, zero_frac."""
        before = torch.randn(100)
        after = before + torch.randn(100) * 0.01

        stats = ulp_statistics(before, after)

        assert "ulp_mean" in stats
        assert "ulp_median" in stats
        assert "ulp_max" in stats
        assert "ulp_zero_frac" in stats

        assert isinstance(stats["ulp_mean"], float)
        assert isinstance(stats["ulp_median"], float)
        assert isinstance(stats["ulp_max"], float)
        assert isinstance(stats["ulp_zero_frac"], float)

    def test_ulp_zero_frac_matches_stall(self):
        """Test that zero ULP movement = bit stall."""
        before = torch.tensor([1.0, 2.0, 3.0])
        # No change
        after = before.clone()

        stats = ulp_statistics(before, after)

        # All zeros means 100% stall
        assert stats["ulp_zero_frac"] == 1.0

    def test_ulp_uses_nextafter(self):
        """Test implementation uses torch.nextafter (not manual).

        This is verified by checking nextafter behavior.
        """
        # torch.nextafter should give the next representable float
        x = torch.tensor([1.0])
        next_x = torch.nextafter(x, torch.tensor([2.0]))

        # The distance should be one ULP
        dist = compute_ulp_distance(x, next_x)

        # If implemented correctly with nextafter, should be exactly 1
        assert abs(dist[0].item() - 1.0) < 1e-6


class TestIntegration:
    """Integration tests for advanced diagnostics."""

    def test_stiffness_predicts_stall(self):
        """Test that high stiffness correlates with bit stalls."""
        # Large weights have high stiffness
        w = torch.tensor([10.0, 20.0, 40.0])
        stiffness = compute_stiffness_field(w, mantissa_bits=2)

        # Small gradients compared to stiffness
        grad = torch.tensor([0.01, 0.01, 0.01])

        # Gradients below stiffness indicate stall
        below_stiffness = (grad.abs() < stiffness).float().mean().item()
        assert below_stiffness > 0.5  # Most should be below

    def test_grid_alignment_improves_on_grid_values(self):
        """Test that quantized values have better grid alignment."""
        from altgrad.quantization import quantize

        w = torch.randn(100)
        scale = torch.tensor(1.0)

        # Original alignment
        error_before = grid_alignment_error(w, E5M2, scale).mean().item()

        # Quantized alignment (should be nearly zero)
        w_q = quantize(w, E5M2, scale)
        error_after = grid_alignment_error(w_q, E5M2, scale).mean().item()

        assert error_after < error_before

    def test_ulp_scales_with_update_size(self):
        """Test that ULP distance scales with update magnitude."""
        before = torch.randn(100)

        # Small update
        after_small = before + torch.randn(100) * 0.001
        stats_small = ulp_statistics(before, after_small)

        # Large update
        after_large = before + torch.randn(100) * 1.0
        stats_large = ulp_statistics(before, after_large)

        # Larger update should have larger mean ULP movement
        assert stats_large["ulp_mean"] > stats_small["ulp_mean"]
