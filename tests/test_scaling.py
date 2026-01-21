"""Tests for per-tensor dynamic scaling with amax history."""

import pytest
import torch

from altgrad.quantization import E0M7, E1M6, E3M4, E5M2, E7M0
from altgrad.quantization.scaling import AmaxHistory, ScalingConfig, compute_scale


class TestAmaxHistory:
    """Tests for AmaxHistory class."""

    def test_init_valid(self):
        """Test initialization with valid parameters."""
        history = AmaxHistory(history_len=16)
        assert len(history) == 0
        assert history.get_amax() == 1.0  # Default when empty

    def test_init_invalid(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError, match="history_len must be positive"):
            AmaxHistory(history_len=0)
        with pytest.raises(ValueError, match="history_len must be positive"):
            AmaxHistory(history_len=-1)

    def test_update_single(self):
        """Test updating with a single tensor."""
        history = AmaxHistory(history_len=4)
        x = torch.tensor([1.0, -2.5, 3.0])
        history.update(x)

        assert len(history) == 1
        assert history.get_amax() == 3.0

    def test_update_multiple(self):
        """Test updating with multiple tensors."""
        history = AmaxHistory(history_len=4)

        history.update(torch.tensor([1.0, -2.0]))
        history.update(torch.tensor([3.0, -1.0]))
        history.update(torch.tensor([0.5, -4.0]))

        assert len(history) == 3
        # Max of [2.0, 3.0, 4.0] = 4.0
        assert history.get_amax() == 4.0

    def test_history_overflow(self):
        """Test that history respects max length."""
        history = AmaxHistory(history_len=3)

        history.update(torch.tensor([1.0]))   # amax=1.0
        history.update(torch.tensor([2.0]))   # amax=2.0
        history.update(torch.tensor([3.0]))   # amax=3.0
        history.update(torch.tensor([0.5]))   # amax=0.5, drops 1.0

        assert len(history) == 3
        # Max of [2.0, 3.0, 0.5] = 3.0
        assert history.get_amax() == 3.0

        history.update(torch.tensor([0.1]))   # amax=0.1, drops 2.0
        # Max of [3.0, 0.5, 0.1] = 3.0
        assert history.get_amax() == 3.0

        history.update(torch.tensor([0.05]))  # amax=0.05, drops 3.0
        # Max of [0.5, 0.1, 0.05] = 0.5
        assert history.get_amax() == 0.5

    def test_reset(self):
        """Test resetting history."""
        history = AmaxHistory(history_len=4)
        history.update(torch.tensor([5.0]))
        history.update(torch.tensor([10.0]))

        assert len(history) == 2
        assert history.get_amax() == 10.0

        history.reset()
        assert len(history) == 0
        assert history.get_amax() == 1.0

    def test_multidimensional_tensors(self):
        """Test with multidimensional tensors."""
        history = AmaxHistory(history_len=4)

        x = torch.randn(10, 20, 30)
        x[5, 10, 15] = 100.0  # Set max value
        history.update(x)

        assert history.get_amax() == 100.0

    def test_empty_history(self):
        """Test behavior with empty history."""
        history = AmaxHistory(history_len=4)
        assert history.get_amax() == 1.0  # Default scale


class TestComputeScale:
    """Tests for compute_scale function."""

    def test_e5m2_scale(self):
        """Test scale computation for E5M2 format."""
        # E5M2 max = 57344
        scale = compute_scale(amax=100.0, format=E5M2)
        expected = 100.0 / 57344.0
        assert abs(scale - expected) < 1e-10

    def test_e3m4_scale(self):
        """Test scale computation for E3M4 format."""
        # E3M4 max = 124 (with bias=1)
        scale = compute_scale(amax=10.0, format=E3M4)
        expected = 10.0 / 124.0
        assert abs(scale - expected) < 1e-10

    def test_e1m6_scale(self):
        """Test scale computation for E1M6 format."""
        # E1M6 max ≈ 1.96875
        scale = compute_scale(amax=1.0, format=E1M6)
        expected = 1.0 / E1M6.max_representable_value
        assert abs(scale - expected) < 1e-10

    def test_e0m7_scale(self):
        """Test scale computation for E0M7 format."""
        # E0M7 max = 127/128 ≈ 0.9921875
        scale = compute_scale(amax=0.5, format=E0M7)
        expected = 0.5 / E0M7.max_representable_value
        assert abs(scale - expected) < 1e-10

    def test_e7m0_scale(self):
        """Test scale computation for E7M0 format."""
        # E7M0 max = 2^63
        scale = compute_scale(amax=1e10, format=E7M0)
        expected = 1e10 / E7M0.max_representable_value
        assert abs(scale - expected) < 1e-10

    def test_scale_min_clamping(self):
        """Test that scale is clamped to minimum."""
        scale = compute_scale(amax=0.0, format=E5M2, scale_min=1e-10)
        assert scale == 1e-10

        scale = compute_scale(amax=1e-15, format=E5M2, scale_min=1e-10)
        assert scale == 1e-10

    def test_scale_min_custom(self):
        """Test custom scale_min parameter."""
        scale = compute_scale(amax=0.0, format=E5M2, scale_min=1e-5)
        assert scale == 1e-5


class TestScalingConfig:
    """Tests for ScalingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ScalingConfig()
        assert config.history_len == 16
        assert config.scale_min == 1e-10

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScalingConfig(history_len=32, scale_min=1e-8)
        assert config.history_len == 32
        assert config.scale_min == 1e-8


class TestIntegration:
    """Integration tests for scaling workflow."""

    def test_typical_workflow(self):
        """Test typical scaling workflow over multiple batches."""
        history = AmaxHistory(history_len=4)

        # Simulate batches with varying ranges
        batches = [
            torch.randn(100) * 5,   # amax ≈ 15-20
            torch.randn(100) * 10,  # amax ≈ 30-40
            torch.randn(100) * 2,   # amax ≈ 6-8
        ]

        scales = []
        for batch in batches:
            history.update(batch)
            scale = compute_scale(history.get_amax(), E5M2)
            scales.append(scale)

        # Scales should be positive and reasonable
        assert all(s > 0 for s in scales)
        # Later scales should consider history
        assert len(history) == 3

    def test_all_zeros(self):
        """Test edge case: all zero tensors."""
        history = AmaxHistory(history_len=4)

        for _ in range(3):
            history.update(torch.zeros(10))

        # Should return scale_min to prevent division by zero
        scale = compute_scale(history.get_amax(), E5M2)
        assert scale == 1e-10  # Default scale_min
