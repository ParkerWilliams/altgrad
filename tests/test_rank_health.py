"""Tests for rank health monitoring."""
import pytest
import torch
import torch.nn as nn
from altgrad.quantization.rank_health import (
    compute_stable_rank, compute_effective_rank,
    RankTrendDetector, RankHealthMonitor
)
from altgrad.training.metrics import compute_rank_stats


class TestStableRank:
    def test_identity_matrix_full_rank(self):
        """Identity matrix has stable rank equal to dimension."""
        eye = torch.eye(10)
        sr = compute_stable_rank(eye)
        assert abs(sr - 10.0) < 0.1  # Should be exactly 10

    def test_rank_one_matrix(self):
        """Rank-1 matrix has stable rank 1."""
        u = torch.randn(10, 1)
        v = torch.randn(1, 5)
        rank1 = u @ v  # Outer product
        sr = compute_stable_rank(rank1)
        assert abs(sr - 1.0) < 0.1

    def test_random_matrix_intermediate_rank(self):
        """Random matrix has intermediate stable rank."""
        w = torch.randn(64, 128)
        sr = compute_stable_rank(w)
        assert 1.0 <= sr <= 64.0  # Between 1 and min(m,n)

    def test_handles_1d_tensor(self):
        """1D tensor returns numel."""
        v = torch.randn(10)
        sr = compute_stable_rank(v)
        assert sr == 10.0

    def test_handles_zero_matrix(self):
        """Zero matrix returns min(m,n)."""
        zero = torch.zeros(10, 5)
        sr = compute_stable_rank(zero)
        assert sr == 5.0  # min(10, 5)

    def test_handles_conv_weights(self):
        """Reshapes conv weights (4D) to 2D for SVD."""
        conv = torch.randn(32, 16, 3, 3)  # (out_ch, in_ch, kH, kW)
        sr = compute_stable_rank(conv)
        # Should be between 1 and min(32, 16*3*3=144) = 32
        assert 1.0 <= sr <= 32.0


class TestEffectiveRank:
    def test_identity_matrix_full_rank(self):
        """Identity matrix has effective rank equal to dimension."""
        eye = torch.eye(10)
        er = compute_effective_rank(eye)
        assert abs(er - 10.0) < 0.1

    def test_rank_one_matrix(self):
        """Rank-1 matrix has effective rank close to 1."""
        u = torch.randn(10, 1)
        v = torch.randn(1, 5)
        rank1 = u @ v
        er = compute_effective_rank(rank1)
        assert er < 2.0  # Should be close to 1

    def test_effective_rank_bounded(self):
        """Effective rank is bounded by matrix dimensions."""
        w = torch.randn(32, 64)
        er = compute_effective_rank(w)
        assert 1.0 <= er <= 32.0

    def test_handles_1d_tensor(self):
        """1D tensor returns 1.0."""
        v = torch.randn(10)
        er = compute_effective_rank(v)
        assert er == 1.0

    def test_handles_zero_matrix(self):
        """Zero matrix returns 1.0 (degenerate case)."""
        zero = torch.zeros(10, 5)
        er = compute_effective_rank(zero)
        assert er == 1.0


class TestRankTrendDetector:
    def test_no_warning_during_warmup(self):
        """No warnings during warmup window."""
        detector = RankTrendDetector(window=10)
        for i in range(10):
            warning = detector.update(10.0 - i * 0.5)  # Dropping rank
            assert warning is None  # Still in warmup

    def test_warning_on_sustained_drop(self):
        """Warning issued on sustained rank drop after warmup."""
        detector = RankTrendDetector(window=10, threshold_pct=0.2, alpha=0.3)
        # Warmup with stable values
        for _ in range(15):
            detector.update(10.0)
        # Now drop significantly
        warning = None
        for _ in range(20):
            warning = detector.update(5.0)
            if warning:
                break
        assert warning is not None
        assert "WARN" in warning or "drop" in warning.lower()

    def test_no_warning_on_stable_values(self):
        """No warning when values are stable."""
        detector = RankTrendDetector(window=5, threshold_pct=0.2)
        for _ in range(20):
            warning = detector.update(10.0)
        assert warning is None

    def test_reset_clears_state(self):
        """reset() clears all tracking state."""
        detector = RankTrendDetector()
        for _ in range(5):
            detector.update(10.0)
        detector.reset()
        assert detector.ema is None
        assert detector.step_count == 0
        assert detector.initial_ema is None


class TestRankHealthMonitor:
    def test_compute_layer_ranks(self):
        """compute_layer_ranks returns valid metrics for model."""
        model = nn.Linear(64, 32)
        monitor = RankHealthMonitor()
        ranks = monitor.compute_layer_ranks(model)

        assert len(ranks) == 1  # One weight matrix
        for name, metrics in ranks.items():
            assert "stable_rank" in metrics
            assert "effective_rank" in metrics
            assert "spectral_norm" in metrics

    def test_skips_1d_params(self):
        """1D parameters (biases) are skipped."""
        model = nn.Linear(64, 32, bias=True)
        monitor = RankHealthMonitor()
        ranks = monitor.compute_layer_ranks(model)

        # Should only have weight, not bias
        assert len(ranks) == 1
        assert "weight" in list(ranks.keys())[0]

    def test_multi_layer_model(self):
        """Works with multi-layer models."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.Linear(16, 10)
        )
        monitor = RankHealthMonitor()
        ranks = monitor.compute_layer_ranks(model)

        # 3 Linear weights + 1 LayerNorm weight (but LN is 1D, so just 3)
        assert len(ranks) == 3

    def test_check_warnings_creates_detectors(self):
        """check_warnings creates detectors on first call."""
        model = nn.Linear(64, 32)
        monitor = RankHealthMonitor()
        ranks = monitor.compute_layer_ranks(model)

        warnings = monitor.check_warnings(ranks)

        assert len(monitor._detectors) == 1
        assert warnings == []  # No warning on first call


class TestComputeRankStats:
    def test_returns_aggregate_stats(self):
        """compute_rank_stats returns per-layer and aggregate metrics."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.Linear(32, 10)
        )
        stats = compute_rank_stats(model)

        assert "stable_rank/mean" in stats
        assert "effective_rank/mean" in stats
        assert "stable_rank/min" in stats
        assert "effective_rank/min" in stats

    def test_returns_per_layer_stats(self):
        """compute_rank_stats returns per-layer metrics."""
        model = nn.Linear(64, 32)
        stats = compute_rank_stats(model)

        # Should have stable_rank/weight and effective_rank/weight
        per_layer_keys = [k for k in stats if "/" in k and "mean" not in k and "min" not in k]
        assert len(per_layer_keys) == 2

    def test_skips_1d_params(self):
        """1D parameters are not included."""
        model = nn.Linear(64, 32, bias=True)
        stats = compute_rank_stats(model)

        # Should not have bias in any key
        assert not any("bias" in k for k in stats)

    def test_empty_model(self):
        """Empty model returns zero aggregates."""
        model = nn.Module()
        stats = compute_rank_stats(model)

        assert stats["stable_rank/mean"] == 0.0
        assert stats["effective_rank/mean"] == 0.0
        assert stats["stable_rank/min"] == 0.0
        assert stats["effective_rank/min"] == 0.0


class TestClassifierSpecificThresholds:
    """Tests for classifier-specific threshold behavior."""

    def test_get_threshold_for_layer_critical(self):
        """Critical layers get stricter threshold."""
        monitor = RankHealthMonitor(
            warn_threshold=0.3, critical_threshold_multiplier=0.5
        )
        # lm_head and c_proj are default critical layers
        assert monitor.get_threshold_for_layer("lm_head.weight") == 0.15
        assert monitor.get_threshold_for_layer("c_proj.weight") == 0.15

    def test_get_threshold_for_layer_non_critical(self):
        """Non-critical layers get standard threshold."""
        monitor = RankHealthMonitor(
            warn_threshold=0.3, critical_threshold_multiplier=0.5
        )
        # These are not critical layers
        assert monitor.get_threshold_for_layer("encoder.layer.0.weight") == 0.3
        assert monitor.get_threshold_for_layer("mlp.fc1.weight") == 0.3

    def test_critical_threshold_multiplier_custom(self):
        """Custom multiplier changes critical layer threshold."""
        monitor = RankHealthMonitor(
            warn_threshold=0.3, critical_threshold_multiplier=0.25
        )
        # 0.3 * 0.25 = 0.075
        assert monitor.get_threshold_for_layer("lm_head.weight") == 0.075

    def test_check_warnings_uses_stricter_threshold(self):
        """check_warnings() uses per-layer threshold from get_threshold_for_layer()."""
        # Create monitor with specific thresholds
        monitor = RankHealthMonitor(
            log_interval=5,
            warn_threshold=0.3,
            critical_threshold_multiplier=0.5,
        )

        # Simulate rank metrics for two layers
        # Initial rank values (warmup phase)
        initial_ranks = {
            "lm_head.weight": {"stable_rank": 10.0, "effective_rank": 10.0, "spectral_norm": 1.0},
            "encoder.weight": {"stable_rank": 10.0, "effective_rank": 10.0, "spectral_norm": 1.0},
        }

        # Warmup: 5 calls to establish baseline
        for _ in range(6):
            monitor.check_warnings(initial_ranks)

        # Now simulate 20% drop (between 0.15 and 0.30 thresholds)
        # lm_head threshold is 0.15, should warn at 20% drop
        # encoder threshold is 0.30, should NOT warn at 20% drop
        dropped_ranks = {
            "lm_head.weight": {"stable_rank": 8.0, "effective_rank": 8.0, "spectral_norm": 1.0},
            "encoder.weight": {"stable_rank": 8.0, "effective_rank": 8.0, "spectral_norm": 1.0},
        }

        # Feed dropped values until warning triggers
        warnings = []
        for _ in range(20):
            warnings.extend(monitor.check_warnings(dropped_ranks))

        # lm_head should warn (20% drop > 15% threshold)
        lm_head_warnings = [w for w in warnings if "lm_head" in w]
        encoder_warnings = [w for w in warnings if "encoder" in w]

        assert len(lm_head_warnings) > 0, "lm_head should warn at 20% drop (threshold 15%)"
        assert len(encoder_warnings) == 0, "encoder should NOT warn at 20% drop (threshold 30%)"

    def test_custom_critical_layers(self):
        """Custom critical_layers list overrides defaults."""
        monitor = RankHealthMonitor(
            warn_threshold=0.3,
            critical_layers=["classifier", "output"],
            critical_threshold_multiplier=0.5,
        )

        # Custom critical layers get stricter threshold
        assert monitor.get_threshold_for_layer("classifier.weight") == 0.15
        assert monitor.get_threshold_for_layer("output.weight") == 0.15

        # Default critical layers (lm_head) should NOT get stricter threshold
        assert monitor.get_threshold_for_layer("lm_head.weight") == 0.3
