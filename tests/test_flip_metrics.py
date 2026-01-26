"""Tests for weight flip metrics."""
import pytest
import torch
from altgrad.quantization.flip_metrics import (
    WeightFlipTracker,
    compute_flip_rate,
    compute_stall_ratio,
)
from altgrad.quantization.formats import E5M2, E3M4


class TestComputeFlipRate:
    def test_identical_tensors_zero_rate(self):
        """Identical quantized tensors have zero flip rate."""
        q = torch.tensor([1.0, 2.0, 3.0])
        assert compute_flip_rate(q, q.clone()) == 0.0

    def test_all_different_full_rate(self):
        """All different values have 100% flip rate."""
        q1 = torch.tensor([1.0, 2.0, 3.0])
        q2 = torch.tensor([4.0, 5.0, 6.0])
        assert compute_flip_rate(q1, q2) == 1.0

    def test_partial_flips(self):
        """Partial changes give correct rate."""
        q1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        q2 = torch.tensor([1.0, 2.0, 5.0, 6.0])  # 2 of 4 changed
        assert compute_flip_rate(q1, q2) == 0.5

    def test_empty_tensors(self):
        """Empty tensors return zero rate."""
        q1 = torch.tensor([])
        q2 = torch.tensor([])
        assert compute_flip_rate(q1, q2) == 0.0

    def test_shape_mismatch_raises(self):
        """Shape mismatch raises ValueError."""
        q1 = torch.tensor([1.0, 2.0, 3.0])
        q2 = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_flip_rate(q1, q2)


class TestWeightFlipTracker:
    def test_no_change_zero_flips(self):
        """Unchanged weights produce zero flips."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(1.0)

        tracker.snapshot_pre_step("layer1", w, E5M2, scale)
        flips = tracker.compute_flips_post_step("layer1", w, E5M2, scale)
        assert flips == 0

    def test_changed_weights_count_flips(self):
        """Changed weights are counted as flips."""
        tracker = WeightFlipTracker()
        w1 = torch.tensor([1.0, 2.0, 3.0])
        w2 = torch.tensor([10.0, 20.0, 30.0])  # All different after quantization
        scale = torch.tensor(1.0)

        tracker.snapshot_pre_step("layer1", w1, E5M2, scale)
        flips = tracker.compute_flips_post_step("layer1", w2, E5M2, scale)
        assert flips == 3

    def test_flip_rates_correct(self):
        """get_flip_rates returns correct per-layer rates."""
        tracker = WeightFlipTracker()
        w = torch.randn(100)
        scale = torch.tensor(1.0)

        tracker.snapshot_pre_step("layer1", w, E5M2, scale)
        tracker.compute_flips_post_step("layer1", w + 0.5, E5M2, scale)

        rates = tracker.get_flip_rates()
        assert "layer1" in rates
        assert 0.0 <= rates["layer1"] <= 1.0

    def test_reset_clears_state(self):
        """reset() clears all tracking state."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0])
        scale = torch.tensor(1.0)

        tracker.snapshot_pre_step("layer1", w, E5M2, scale)
        tracker.compute_flips_post_step("layer1", w + 1.0, E5M2, scale)

        tracker.reset()
        assert tracker.get_flip_counts() == {}
        assert tracker.get_flip_rates() == {}

    def test_multiple_layers(self):
        """Tracker handles multiple layers independently."""
        tracker = WeightFlipTracker()
        w1 = torch.tensor([1.0, 2.0])
        w2 = torch.tensor([3.0, 4.0, 5.0])
        scale = torch.tensor(1.0)

        tracker.snapshot_pre_step("layer1", w1, E5M2, scale)
        tracker.snapshot_pre_step("layer2", w2, E3M4, scale)

        # layer1: no change
        flips1 = tracker.compute_flips_post_step("layer1", w1, E5M2, scale)
        # layer2: all change
        flips2 = tracker.compute_flips_post_step("layer2", w2 * 10.0, E3M4, scale)

        assert flips1 == 0
        assert flips2 == 3

        counts = tracker.get_flip_counts()
        assert counts["layer1"] == 0
        assert counts["layer2"] == 3

    def test_cumulative_counts(self):
        """Flip counts accumulate over multiple steps."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0])
        scale = torch.tensor(1.0)

        # Step 1: 2 flips
        tracker.snapshot_pre_step("layer1", w, E5M2, scale)
        tracker.compute_flips_post_step("layer1", w * 10.0, E5M2, scale)

        # Step 2: 2 more flips
        tracker.snapshot_pre_step("layer1", w * 10.0, E5M2, scale)
        tracker.compute_flips_post_step("layer1", w, E5M2, scale)

        # Total: 4 flips
        assert tracker.get_flip_counts()["layer1"] == 4
        # Rate: 4 flips / 2 weights = 2.0
        assert tracker.get_flip_rates()["layer1"] == 2.0

    def test_missing_snapshot_raises(self):
        """compute_flips_post_step raises if snapshot not taken."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0])
        scale = torch.tensor(1.0)

        with pytest.raises(KeyError, match="No pre-step snapshot"):
            tracker.compute_flips_post_step("missing_layer", w, E5M2, scale)

    def test_snapshot_cleans_up_after_compute(self):
        """Snapshot is removed after computing flips."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0])
        scale = torch.tensor(1.0)

        tracker.snapshot_pre_step("layer1", w, E5M2, scale)
        assert "layer1" in tracker.prev_quantized

        tracker.compute_flips_post_step("layer1", w, E5M2, scale)
        assert "layer1" not in tracker.prev_quantized


class TestComputeStallRatio:
    """Tests for compute_stall_ratio function."""

    def test_compute_stall_ratio_basic(self):
        """Test basic stall ratio calculations."""
        # 10 flips from 100 updates = 90% stall
        assert compute_stall_ratio(10, 100) == 0.9
        # 100 flips from 100 updates = 0% stall (all flip)
        assert compute_stall_ratio(100, 100) == 0.0
        # 0 flips from 100 updates = 100% stall (no flips)
        assert compute_stall_ratio(0, 100) == 1.0
        # 0 updates = 0% stall (no gradient = no stall by definition)
        assert compute_stall_ratio(0, 0) == 0.0


class TestTrackerUpdateCounts:
    """Tests for WeightFlipTracker update tracking."""

    def test_tracker_update_counts(self):
        """Test that update counts track non-zero gradients."""
        tracker = WeightFlipTracker()
        w = torch.randn(100)
        scale = torch.tensor(1.0)

        # Create gradient with exactly 50 non-zero elements
        grad = torch.zeros(100)
        grad[:50] = torch.randn(50)  # First 50 non-zero

        tracker.snapshot_pre_step("layer1", w, E5M2, scale, grad=grad)
        counts = tracker.get_update_counts()
        assert counts["layer1"] == 50

    def test_tracker_stall_ratios(self):
        """Test stall ratio calculation from tracker state."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor(1.0)

        # Create gradient with all non-zero (4 updates)
        grad = torch.ones(4)

        tracker.snapshot_pre_step("layer1", w, E5M2, scale, grad=grad)
        # Modify weights to cause 2 flips (half the elements)
        w_after = w.clone()
        w_after[:2] = w[:2] * 10.0  # Change first 2 significantly

        flips = tracker.compute_flips_post_step("layer1", w_after, E5M2, scale)
        ratios = tracker.get_stall_ratios()

        # 2 flips / 4 updates = 50% stall rate
        assert "layer1" in ratios
        # Stall = 1 - (flips / updates), so if 2 flips from 4 updates => 0.5 stall
        assert ratios["layer1"] == 1.0 - (flips / 4)

    def test_tracker_stall_ratio_no_updates(self):
        """Test stall ratio with zero gradient returns 0.0 (not NaN)."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(1.0)

        # Zero gradient = no updates
        grad = torch.zeros(3)

        tracker.snapshot_pre_step("layer1", w, E5M2, scale, grad=grad)
        tracker.compute_flips_post_step("layer1", w, E5M2, scale)

        ratios = tracker.get_stall_ratios()
        # No updates = 0% stall by definition
        assert ratios["layer1"] == 0.0

    def test_tracker_reset_clears_updates(self):
        """Test that reset() clears update_counts."""
        tracker = WeightFlipTracker()
        w = torch.tensor([1.0, 2.0])
        scale = torch.tensor(1.0)
        grad = torch.ones(2)

        tracker.snapshot_pre_step("layer1", w, E5M2, scale, grad=grad)
        assert tracker.get_update_counts() == {"layer1": 2}

        tracker.reset()
        assert tracker.get_update_counts() == {}
