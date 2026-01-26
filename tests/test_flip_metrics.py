"""Tests for weight flip metrics."""
import pytest
import torch
from altgrad.quantization.flip_metrics import WeightFlipTracker, compute_flip_rate
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
