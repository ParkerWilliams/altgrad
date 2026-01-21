"""Tests for training infrastructure modules.

Tests configuration, metrics, checkpoint, and W&B callback functionality.
W&B tests use mocking to avoid actual API calls.
"""

import math
import os
import random
import tempfile
from dataclasses import asdict
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from altgrad.training.config import TrainConfig, load_config, save_config
from altgrad.training.metrics import (
    compute_gradient_stats,
    compute_stability_metrics,
    gradient_cosine_similarity,
)
from altgrad.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointManager,
)


# =============================================================================
# Config tests
# =============================================================================


class TestConfig:
    """Test configuration dataclass and YAML I/O."""

    def test_config_defaults(self):
        """TrainConfig has sensible defaults."""
        config = TrainConfig()

        # Model defaults
        assert config.n_layer == 6
        assert config.n_head == 6
        assert config.n_embd == 384
        assert config.vocab_size == 50304
        assert config.dropout == 0.0

        # Training defaults
        assert config.batch_size == 64
        assert config.learning_rate == 3e-4
        assert config.grad_clip == 1.0

        # Quantization defaults
        assert config.use_fp8 is False
        assert config.fp8_format == "E5M2"

        # Checkpointing defaults
        assert config.checkpoint_interval == 100
        assert config.max_checkpoints == 3

        # Stability thresholds
        assert config.nan_patience == 10
        assert config.bit_stall_threshold == 0.5
        assert config.dead_neuron_threshold == 1e-8

    def test_config_yaml_roundtrip(self, tmp_path):
        """save_config then load_config preserves values."""
        config = TrainConfig(
            n_layer=12,
            n_head=8,
            learning_rate=1e-4,
            use_fp8=True,
            fp8_format="E3M4",
            tags=["test", "experiment"],
            seed=123,
        )

        yaml_path = str(tmp_path / "config.yaml")
        save_config(config, yaml_path)
        loaded = load_config(yaml_path)

        # All fields preserved
        assert loaded.n_layer == 12
        assert loaded.n_head == 8
        assert loaded.learning_rate == 1e-4
        assert loaded.use_fp8 is True
        assert loaded.fp8_format == "E3M4"
        assert loaded.tags == ["test", "experiment"]
        assert loaded.seed == 123

        # Defaults also preserved
        assert loaded.n_embd == 384
        assert loaded.batch_size == 64

    def test_config_fp8_options(self):
        """use_fp8 and fp8_format work correctly."""
        # FP32 baseline
        config_fp32 = TrainConfig(use_fp8=False)
        assert config_fp32.use_fp8 is False
        assert config_fp32.use_shadow is False

        # FP8 E5M2
        config_e5m2 = TrainConfig(use_fp8=True, fp8_format="E5M2")
        assert config_e5m2.use_fp8 is True
        assert config_e5m2.fp8_format == "E5M2"

        # FP8 E3M4 with shadow
        config_e3m4 = TrainConfig(use_fp8=True, fp8_format="E3M4", use_shadow=True)
        assert config_e3m4.fp8_format == "E3M4"
        assert config_e3m4.use_shadow is True


# =============================================================================
# Metrics tests
# =============================================================================


class TestGradientStats:
    """Test gradient statistics computation."""

    def test_gradient_stats_shapes(self):
        """Returns dict with expected keys."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        model(x).sum().backward()

        stats = compute_gradient_stats(model)

        # Per-layer keys
        assert "grad_norm_l2/weight" in stats
        assert "grad_norm_l2/bias" in stats
        assert "grad_norm_linf/weight" in stats
        assert "dead_neuron_frac/weight" in stats
        assert "grad_snr/weight" in stats

        # Aggregate keys
        assert "grad_norm_l2/mean" in stats
        assert "grad_norm_l2/min" in stats
        assert "grad_norm_linf/max" in stats
        assert "dead_neuron_frac/mean" in stats
        assert "grad_snr/mean" in stats
        assert "grad_snr/min" in stats

    def test_gradient_l2_norm(self):
        """Correct L2 norm computation."""
        model = nn.Linear(4, 2, bias=False)
        # Set gradient to known values
        model.weight.grad = torch.ones(2, 4)  # 8 elements, all 1.0

        stats = compute_gradient_stats(model)

        # L2 norm of [1,1,1,1,1,1,1,1] = sqrt(8) ≈ 2.828
        expected = math.sqrt(8)
        assert abs(stats["grad_norm_l2/weight"] - expected) < 1e-5

    def test_gradient_dead_fraction(self):
        """Correctly identifies gradients below threshold."""
        model = nn.Linear(4, 2, bias=False)
        # Half zeros, half ones
        model.weight.grad = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

        stats = compute_gradient_stats(model, threshold=0.5)

        # 4 out of 8 elements are below 0.5
        assert abs(stats["dead_neuron_frac/weight"] - 0.5) < 1e-5

    def test_gradient_snr(self):
        """Signal-to-noise ratio computed correctly."""
        model = nn.Linear(100, 1, bias=False)
        # Constant gradient = high SNR (mean/std would be inf for constant)
        # But we add tiny noise
        model.weight.grad = torch.ones(1, 100) + torch.randn(1, 100) * 0.01

        stats = compute_gradient_stats(model)

        # Mean ≈ 1.0, std ≈ 0.01 -> SNR ≈ 100
        # Actually SNR = |mean| / std
        assert stats["grad_snr/weight"] > 50  # Should be high

    def test_cosine_similarity_identical(self):
        """Identical gradients give similarity ~1.0."""
        model_a = nn.Linear(10, 5)
        model_b = nn.Linear(10, 5)

        # Same input
        x = torch.randn(4, 10)
        model_a(x).sum().backward()
        model_b(x).sum().backward()

        # Copy gradients to make identical
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            pb.grad = pa.grad.clone()

        sim = gradient_cosine_similarity(model_a, model_b)

        # Should be exactly 1.0
        assert abs(sim["grad_cos_sim/weight"] - 1.0) < 1e-5
        assert abs(sim["grad_cos_sim/bias"] - 1.0) < 1e-5
        assert abs(sim["grad_cos_sim/mean"] - 1.0) < 1e-5

    def test_cosine_similarity_opposite(self):
        """Opposite gradients give similarity ~-1.0."""
        model_a = nn.Linear(10, 5)
        model_b = nn.Linear(10, 5)

        x = torch.randn(4, 10)
        model_a(x).sum().backward()
        model_b(x).sum().backward()

        # Negate gradients in model_b
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            pb.grad = -pa.grad.clone()

        sim = gradient_cosine_similarity(model_a, model_b)

        # Should be -1.0
        assert abs(sim["grad_cos_sim/weight"] - (-1.0)) < 1e-5
        assert abs(sim["grad_cos_sim/mean"] - (-1.0)) < 1e-5


class TestStabilityMetrics:
    """Test stability metrics computation."""

    def test_stability_no_issues(self):
        """Normal model has no NaN/Inf."""
        model = nn.Linear(10, 5)
        metrics = compute_stability_metrics(model)

        assert metrics["param_nan_count"] == 0
        assert metrics["param_inf_count"] == 0

    def test_stability_with_detector(self):
        """Includes bit_stall_rate when detector provided."""
        from altgrad.quantization import BitStallDetector

        model = nn.Linear(10, 5)
        detector = BitStallDetector()
        # Manually set some stats
        detector.stall_count = 10
        detector.total_count = 100

        metrics = compute_stability_metrics(model, detector=detector)

        assert "bit_stall_rate" in metrics
        assert abs(metrics["bit_stall_rate"] - 0.1) < 1e-5


# =============================================================================
# Checkpoint tests
# =============================================================================


class TestCheckpoint:
    """Test checkpoint save/load functionality."""

    def test_checkpoint_save_load(self, tmp_path):
        """Model state restored correctly."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainConfig(n_layer=4, seed=42)

        # Modify model state
        model.weight.data.fill_(1.5)
        optimizer.step()  # Create optimizer state

        ckpt_path = str(tmp_path / "test.pt")
        save_checkpoint(ckpt_path, model, optimizer, None, step=50, config=config)

        # Create fresh model and load
        model2 = nn.Linear(10, 5)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        step, loaded_config, quant = load_checkpoint(ckpt_path, model2, optimizer2)

        assert step == 50
        assert loaded_config["n_layer"] == 4
        assert torch.allclose(model2.weight.data, model.weight.data)

    def test_checkpoint_rng_state(self, tmp_path):
        """RNG state restored (same random sequence after load)."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainConfig()

        # Set known RNG state
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        ckpt_path = str(tmp_path / "rng.pt")
        save_checkpoint(ckpt_path, model, optimizer, None, step=0, config=config)

        # Generate some random numbers
        r1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        # Load checkpoint (restores RNG state)
        load_checkpoint(ckpt_path, model, optimizer)

        # Same random numbers should be generated
        r2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert r1 == r2
        assert n1 == n2
        assert t1 == t2

    def test_checkpoint_quantization_state(self, tmp_path):
        """Custom quantization_state preserved."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainConfig()

        quant_state = {
            "amax_history": [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
            "scale_factors": {"layer1": 0.5, "layer2": 0.25},
        }

        ckpt_path = str(tmp_path / "quant.pt")
        save_checkpoint(
            ckpt_path, model, optimizer, None, step=100, config=config,
            quantization_state=quant_state
        )

        _, _, loaded_quant = load_checkpoint(ckpt_path, model, optimizer)

        assert loaded_quant["amax_history"] == quant_state["amax_history"]
        assert loaded_quant["scale_factors"] == quant_state["scale_factors"]


class TestCheckpointManager:
    """Test CheckpointManager rotation and best tracking."""

    def test_checkpoint_manager_max_checkpoints(self, tmp_path):
        """Old checkpoints deleted beyond max."""
        manager = CheckpointManager(str(tmp_path), max_checkpoints=2)
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainConfig()

        # Save 4 checkpoints
        manager.save(100, model, optimizer, None, config, val_loss=1.0)
        manager.save(200, model, optimizer, None, config, val_loss=0.9)
        manager.save(300, model, optimizer, None, config, val_loss=0.8)
        manager.save(400, model, optimizer, None, config, val_loss=0.7)

        # Only 2 most recent should exist (plus best)
        assert not os.path.exists(tmp_path / "step_100.pt")
        assert not os.path.exists(tmp_path / "step_200.pt")
        assert os.path.exists(tmp_path / "step_300.pt")
        assert os.path.exists(tmp_path / "step_400.pt")
        assert os.path.exists(tmp_path / "best.pt")

    def test_checkpoint_manager_best_tracking(self, tmp_path):
        """Best checkpoint updated on improvement."""
        manager = CheckpointManager(str(tmp_path), max_checkpoints=3)
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = TrainConfig()

        # Save checkpoints with varying loss
        manager.save(100, model, optimizer, None, config, val_loss=1.0)
        assert manager.best_loss == 1.0

        manager.save(200, model, optimizer, None, config, val_loss=0.8)
        assert manager.best_loss == 0.8

        # Worse loss shouldn't update best
        manager.save(300, model, optimizer, None, config, val_loss=0.9)
        assert manager.best_loss == 0.8  # Unchanged

        # Better loss updates
        manager.save(400, model, optimizer, None, config, val_loss=0.5)
        assert manager.best_loss == 0.5

        # Verify best path exists
        assert manager.best() == str(tmp_path / "best.pt")
        assert os.path.exists(manager.best())


# =============================================================================
# W&B tests (mocked)
# =============================================================================


class TestWandbTracker:
    """Test W&B tracker with mocked wandb module."""

    @mock.patch("altgrad.training.callbacks._wandb", None)
    @mock.patch("altgrad.training.callbacks._get_wandb")
    def test_wandb_tracker_init(self, mock_get_wandb):
        """Initializes with config."""
        # Setup mock
        mock_wandb = mock.MagicMock()
        mock_wandb.init.return_value = mock.MagicMock(id="test-run-123")
        mock_get_wandb.return_value = mock_wandb

        from altgrad.training.callbacks import WandbTracker

        config = TrainConfig(project="test-project", run_name="test-run", tags=["a"])
        tracker = WandbTracker(config)

        # Verify init called with correct args
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["name"] == "test-run"
        assert call_kwargs["tags"] == ["a"]
        assert "config" in call_kwargs

    @mock.patch("altgrad.training.callbacks._wandb", None)
    @mock.patch("altgrad.training.callbacks._get_wandb")
    def test_wandb_log_step(self, mock_get_wandb):
        """Calls wandb.log with explicit step."""
        mock_wandb = mock.MagicMock()
        mock_wandb.init.return_value = mock.MagicMock(id="run-1")
        mock_get_wandb.return_value = mock_wandb

        from altgrad.training.callbacks import WandbTracker

        tracker = WandbTracker(TrainConfig())
        tracker.log_step(42, {"loss": 0.5, "accuracy": 0.9})

        mock_wandb.log.assert_called_once_with(
            {"loss": 0.5, "accuracy": 0.9}, step=42
        )

    @mock.patch("altgrad.training.callbacks._wandb", None)
    @mock.patch("altgrad.training.callbacks._get_wandb")
    def test_wandb_alert_nan(self, mock_get_wandb):
        """Triggers alert after nan_patience exceeded."""
        mock_wandb = mock.MagicMock()
        mock_wandb.init.return_value = mock.MagicMock(id="run-1")
        mock_wandb.AlertLevel = mock.MagicMock()
        mock_wandb.AlertLevel.ERROR = "ERROR"
        mock_get_wandb.return_value = mock_wandb

        from altgrad.training.callbacks import WandbTracker

        config = TrainConfig(nan_patience=3)
        tracker = WandbTracker(config)

        # First NaN: save checkpoint (first occurrence triggers checkpoint)
        action = tracker.check_alerts(1, {"loss": float("nan")}, config)
        assert action == "save_checkpoint"
        assert tracker.consecutive_nan_steps == 1

        # Second NaN: continue (under patience threshold)
        action = tracker.check_alerts(2, {"loss": float("nan")}, config)
        assert action == "continue"  # Under patience, just continue
        assert tracker.consecutive_nan_steps == 2

        # Third NaN: stop (reached nan_patience=3)
        action = tracker.check_alerts(3, {"loss": float("nan")}, config)
        assert action == "stop"
        mock_wandb.alert.assert_called_once()

        # Valid loss resets counter
        tracker.check_alerts(4, {"loss": 0.5}, config)
        assert tracker.consecutive_nan_steps == 0

    @mock.patch("altgrad.training.callbacks._wandb", None)
    @mock.patch("altgrad.training.callbacks._get_wandb")
    def test_wandb_alert_bit_stall(self, mock_get_wandb):
        """Triggers alert on high stall rate."""
        mock_wandb = mock.MagicMock()
        mock_wandb.init.return_value = mock.MagicMock(id="run-1")
        mock_wandb.AlertLevel = mock.MagicMock()
        mock_wandb.AlertLevel.WARN = "WARN"
        mock_get_wandb.return_value = mock_wandb

        from altgrad.training.callbacks import WandbTracker

        config = TrainConfig(bit_stall_threshold=0.3)
        tracker = WandbTracker(config)

        # Below threshold: no alert
        action = tracker.check_alerts(1, {"bit_stall_rate": 0.2}, config)
        assert action == "continue"
        mock_wandb.alert.assert_not_called()

        # Above threshold: alert
        action = tracker.check_alerts(2, {"bit_stall_rate": 0.5}, config)
        assert action == "continue"  # Warn, but don't stop
        mock_wandb.alert.assert_called_once()
        call_kwargs = mock_wandb.alert.call_args.kwargs
        assert "bit stall" in call_kwargs["title"].lower()
