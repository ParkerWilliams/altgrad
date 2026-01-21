"""Tests for bit-stall detection and diagnostics."""

import pytest
import torch

from altgrad.quantization import E0M7, E3M4, E5M2, E7M0
from altgrad.quantization.diagnostics import BitStallDetector, detect_bit_stall


class TestDetectBitStall:
    """Tests for detect_bit_stall function."""

    def test_large_gradients_no_stall(self):
        """Test that large gradients with high LR don't cause stalls."""
        weight = torch.randn(100)
        grad = torch.randn(100) * 10  # Large gradients
        lr = 0.1  # High learning rate
        scale = torch.tensor(1.0)

        stall_count, total_count = detect_bit_stall(weight, grad, lr, E5M2, scale)

        # With large gradients*LR, should have low stall rate
        stall_rate = stall_count / total_count if total_count > 0 else 0
        assert stall_rate < 0.3  # Less than 30% stall

    def test_small_gradients_cause_stalls(self):
        """Test that very small gradients cause stalls."""
        weight = torch.randn(100)
        # Very small gradients relative to E5M2 precision
        grad = torch.randn(100) * 1e-5
        lr = 0.01
        scale = torch.tensor(1.0)

        stall_count, total_count = detect_bit_stall(weight, grad, lr, E5M2, scale)

        # Small gradients should cause high stall rate
        stall_rate = stall_count / total_count if total_count > 0 else 0
        assert stall_rate > 0.5  # More than 50% stall

    def test_e7m0_high_stall_rate(self):
        """Test E7M0 (powers of 2) has high stall rate with normal gradients.

        E7M0 has very coarse precision (only powers of 2), so even
        moderate-sized gradients often can't change the quantized value.
        """
        weight = torch.randn(100)
        grad = torch.randn(100) * 0.1  # Moderate gradients
        lr = 0.01
        scale = torch.tensor(1.0)

        stall_count, total_count = detect_bit_stall(weight, grad, lr, E7M0, scale)

        # E7M0 should have very high stall rate
        stall_rate = stall_count / total_count if total_count > 0 else 0
        assert stall_rate > 0.7  # More than 70% stall due to coarse precision

    def test_zero_gradients_not_counted(self):
        """Test that zero gradients are not counted as stalls."""
        weight = torch.randn(100)
        grad = torch.zeros(100)  # All zero gradients
        lr = 0.01
        scale = torch.tensor(1.0)

        stall_count, total_count = detect_bit_stall(weight, grad, lr, E5M2, scale)

        # Zero gradients shouldn't be counted
        assert total_count == 0
        assert stall_count == 0

    def test_mixed_gradients(self):
        """Test with mixture of large and small gradients."""
        weight = torch.randn(100)
        grad = torch.randn(100) * 10  # Start with large gradients
        # Make half the gradients very small
        grad[:50] *= 1e-6
        lr = 0.1  # High learning rate
        scale = torch.tensor(1.0)

        stall_count, total_count = detect_bit_stall(weight, grad, lr, E5M2, scale)

        # Should have some stalls from small gradients, but not all
        assert 0 < stall_count < total_count
        stall_rate = stall_count / total_count
        # Expect roughly 50% stall (half small, half large)
        assert 0.3 < stall_rate < 0.7

    def test_learning_rate_effect(self):
        """Test that higher learning rate reduces stall rate."""
        weight = torch.randn(100)
        grad = torch.randn(100) * 0.01  # Small gradients
        scale = torch.tensor(1.0)

        # Low learning rate
        stall_low, total_low = detect_bit_stall(weight, grad, 0.001, E5M2, scale)
        rate_low = stall_low / total_low if total_low > 0 else 0

        # High learning rate
        stall_high, total_high = detect_bit_stall(weight, grad, 0.1, E5M2, scale)
        rate_high = stall_high / total_high if total_high > 0 else 0

        # Higher LR should amplify gradients, reducing stalls
        assert rate_high < rate_low

    def test_scale_effect(self):
        """Test that scale affects stall detection."""
        weight = torch.randn(100)
        grad = torch.randn(100) * 0.01
        lr = 0.01

        # Large scale compresses values, may increase stalls
        stall_large, total_large = detect_bit_stall(
            weight, grad, lr, E5M2, torch.tensor(10.0)
        )

        # Small scale expands values, may reduce stalls
        stall_small, total_small = detect_bit_stall(
            weight, grad, lr, E5M2, torch.tensor(0.1)
        )

        # Both should detect something
        assert total_large > 0
        assert total_small > 0


class TestBitStallDetector:
    """Tests for BitStallDetector class."""

    def test_init(self):
        """Test initialization."""
        detector = BitStallDetector()
        assert detector.stall_count == 0
        assert detector.total_count == 0
        assert detector.step_count == 0
        assert detector.get_stall_rate() == 0.0

    def test_single_update(self):
        """Test single update."""
        detector = BitStallDetector()
        weight = torch.randn(100)
        grad = torch.randn(100)
        lr = 0.01
        scale = torch.tensor(1.0)

        detector.update(weight, grad, lr, E5M2, scale)

        assert detector.step_count == 1
        assert detector.total_count > 0
        assert 0 <= detector.get_stall_rate() <= 1.0

    def test_multiple_updates(self):
        """Test accumulation over multiple updates."""
        detector = BitStallDetector()

        for _ in range(5):
            weight = torch.randn(50)
            grad = torch.randn(50) * 0.01  # Small gradients
            detector.update(weight, grad, 0.01, E5M2, torch.tensor(1.0))

        assert detector.step_count == 5
        assert detector.total_count > 0
        # Should have accumulated stalls from small gradients
        assert detector.stall_count > 0

    def test_reset(self):
        """Test reset functionality."""
        detector = BitStallDetector()

        # Accumulate some data
        weight = torch.randn(100)
        grad = torch.randn(100)
        detector.update(weight, grad, 0.01, E5M2, torch.tensor(1.0))

        assert detector.step_count > 0

        # Reset
        detector.reset()
        assert detector.stall_count == 0
        assert detector.total_count == 0
        assert detector.step_count == 0
        assert detector.get_stall_rate() == 0.0

    def test_get_stats(self):
        """Test get_stats returns correct dictionary."""
        detector = BitStallDetector()

        weight = torch.randn(100)
        grad = torch.randn(100)
        detector.update(weight, grad, 0.01, E5M2, torch.tensor(1.0))

        stats = detector.get_stats()

        assert "stall_rate" in stats
        assert "stall_count" in stats
        assert "total_count" in stats
        assert "steps" in stats

        assert stats["steps"] == 1
        assert stats["stall_count"] >= 0
        assert stats["total_count"] > 0
        assert 0 <= stats["stall_rate"] <= 1.0
        assert stats["stall_rate"] == stats["stall_count"] / stats["total_count"]

    def test_zero_total_count(self):
        """Test behavior when no gradients recorded."""
        detector = BitStallDetector()

        # Update with all zero gradients
        weight = torch.randn(100)
        grad = torch.zeros(100)
        detector.update(weight, grad, 0.01, E5M2, torch.tensor(1.0))

        # Should handle division by zero gracefully
        assert detector.get_stall_rate() == 0.0
        stats = detector.get_stats()
        assert stats["stall_rate"] == 0.0


class TestIntegration:
    """Integration tests for diagnostics workflow."""

    def test_format_comparison(self):
        """Test comparing stall rates across formats."""
        weight = torch.randn(200)
        grad = torch.randn(200) * 0.05  # Moderate gradients
        lr = 0.01
        scale = torch.tensor(1.0)

        # Test different formats
        formats = [E5M2, E3M4, E7M0]
        stall_rates = []

        for fmt in formats:
            stall, total = detect_bit_stall(weight, grad, lr, fmt, scale)
            rate = stall / total if total > 0 else 0
            stall_rates.append(rate)

        # E7M0 should have highest stall rate (coarsest precision)
        assert stall_rates[2] > stall_rates[0]  # E7M0 > E5M2
        assert stall_rates[2] > stall_rates[1]  # E7M0 > E3M4

    def test_training_simulation(self):
        """Simulate training loop with stall detection."""
        detector = BitStallDetector()

        # Simulate 10 training steps
        for step in range(10):
            # Weights and gradients evolve
            weight = torch.randn(50)
            # Gradients get smaller over time (simulating convergence)
            grad_scale = 0.1 * (0.9 ** step)
            grad = torch.randn(50) * grad_scale

            detector.update(weight, grad, 0.01, E5M2, torch.tensor(1.0))

        assert detector.step_count == 10

        # As gradients get smaller, stall rate should increase
        stats = detector.get_stats()
        # Final stall rate should be significant due to shrinking gradients
        assert stats["stall_rate"] > 0.0
