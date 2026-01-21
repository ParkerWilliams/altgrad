"""Tests for GPT model, FP32 shadow model, and Trainer.

Tests cover:
  - GPT model architecture and forward pass
  - FP32ShadowModel gradient comparison and SNR metrics
  - Trainer initialization and train_step
"""

import os
import tempfile
from unittest import mock

import pytest
import torch

from altgrad.training.model import GPT, GPTConfig
from altgrad.training.shadow import FP32ShadowModel
from altgrad.training.config import TrainConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tiny_config():
    """Tiny model config for fast tests."""
    return GPTConfig(
        n_layer=1,
        n_head=1,
        n_embd=32,
        block_size=16,
        vocab_size=100,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Tiny GPT model for testing."""
    return GPT(tiny_config)


@pytest.fixture
def sample_batch():
    """Sample input/target batch."""
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    return x, y


@pytest.fixture
def tiny_train_config():
    """Minimal training config for tests."""
    return TrainConfig(
        n_layer=1,
        n_head=1,
        n_embd=32,
        block_size=16,
        vocab_size=100,
        batch_size=2,
        learning_rate=1e-3,
        max_steps=10,
        checkpoint_interval=5,
        eval_interval=5,
        log_interval=1,
        project="",  # Disable W&B
    )


# ============================================================================
# GPT Model Tests
# ============================================================================


class TestGPT:
    """Tests for GPT model."""

    def test_gpt_forward_shape(self, tiny_model, sample_batch):
        """Output shape is (batch, seq_len, vocab_size)."""
        x, _ = sample_batch
        logits, loss = tiny_model(x)

        assert logits.shape == (2, 16, 100)  # (batch, seq, vocab)
        assert loss is None

    def test_gpt_forward_with_targets(self, tiny_model, sample_batch):
        """Returns loss when targets provided."""
        x, y = sample_batch
        logits, loss = tiny_model(x, targets=y)

        assert logits.shape == (2, 16, 100)
        assert loss is not None
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Cross-entropy is positive

    def test_gpt_forward_no_targets(self, tiny_model, sample_batch):
        """Returns None loss when no targets."""
        x, _ = sample_batch
        logits, loss = tiny_model(x, targets=None)

        assert logits.shape == (2, 16, 100)
        assert loss is None

    def test_gpt_causal_mask(self, tiny_config):
        """Future tokens don't affect current predictions."""
        model = GPT(tiny_config)
        model.eval()

        # Create two sequences: identical prefix, different suffix
        x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        x2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99, 99, 99, 99, 99]])

        with torch.no_grad():
            logits1, _ = model(x1)
            logits2, _ = model(x2)

        # Predictions at position 0-7 should be identical (causal mask)
        # because they only see the same prefix tokens
        for i in range(8):
            torch.testing.assert_close(
                logits1[0, i], logits2[0, i],
                msg=f"Position {i} should be identical due to causal mask"
            )

    def test_gpt_configure_optimizers(self, tiny_model):
        """Returns AdamW with correct param groups."""
        optimizer = tiny_model.configure_optimizers(
            weight_decay=0.1,
            learning_rate=1e-3,
            betas=(0.9, 0.95),
            device_type="cpu",
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2

        # First group has weight decay, second doesn't
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        assert decay_group["weight_decay"] == 0.1
        assert no_decay_group["weight_decay"] == 0.0

        # All params should be accounted for
        total_params = sum(len(g["params"]) for g in optimizer.param_groups)
        model_params = sum(1 for p in tiny_model.parameters() if p.requires_grad)
        assert total_params == model_params


# ============================================================================
# FP32 Shadow Model Tests
# ============================================================================


class TestFP32ShadowModel:
    """Tests for FP32 shadow model."""

    def test_shadow_init_copies_weights(self, tiny_model):
        """Shadow has same initial weights."""
        shadow = FP32ShadowModel(tiny_model)

        for (name, p_model), (_, p_shadow) in zip(
            tiny_model.named_parameters(),
            shadow.model.named_parameters(),
        ):
            torch.testing.assert_close(
                p_model.float(), p_shadow,
                msg=f"Weight {name} should be identical after init"
            )

    def test_shadow_sync_weights(self, tiny_model):
        """sync_weights copies current weights."""
        shadow = FP32ShadowModel(tiny_model)

        # Modify model weights
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Weights should now differ
        model_sum = sum(p.sum().item() for p in tiny_model.parameters())
        shadow_sum = sum(p.sum().item() for p in shadow.model.parameters())
        assert model_sum != shadow_sum, "Weights should differ before sync"

        # Sync and verify match
        shadow.sync_weights(tiny_model)

        for (name, p_model), (_, p_shadow) in zip(
            tiny_model.named_parameters(),
            shadow.model.named_parameters(),
        ):
            torch.testing.assert_close(
                p_model.float(), p_shadow,
                msg=f"Weight {name} should match after sync"
            )

    def test_shadow_gradient_similarity_identical(self, tiny_model, sample_batch):
        """Same model gives similarity ~1.0."""
        x, y = sample_batch
        shadow = FP32ShadowModel(tiny_model)

        # Run forward/backward on model
        tiny_model.zero_grad()
        loss = tiny_model(x, y)[1]
        loss.backward()

        # Run forward/backward on shadow (same weights, same input)
        shadow.forward_backward(x, y)

        # Compute similarity
        metrics = shadow.compute_gradient_similarity(tiny_model)

        # Should be very close to 1.0
        assert metrics["grad_cos_sim/mean"] > 0.99

    def test_shadow_gradient_similarity_after_update(self, tiny_model, sample_batch):
        """Similarity changes after training steps diverge models."""
        x, y = sample_batch
        shadow = FP32ShadowModel(tiny_model)

        # Update main model several times (diverge from shadow)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.1)
        for _ in range(10):
            optimizer.zero_grad()
            loss = tiny_model(x, y)[1]
            loss.backward()
            optimizer.step()

        # Run one more step on main model
        tiny_model.zero_grad()
        loss = tiny_model(x, y)[1]
        loss.backward()

        # Shadow still has original weights
        shadow.forward_backward(x, y)

        metrics = shadow.compute_gradient_similarity(tiny_model)

        # Similarity should be lower now (but still positive usually)
        assert metrics["grad_cos_sim/mean"] < 1.0

    def test_shadow_snr_comparison(self, tiny_model, sample_batch):
        """SNR metrics computed for FP8 vs FP32 gradient comparison."""
        x, y = sample_batch
        shadow = FP32ShadowModel(tiny_model)

        # Run forward/backward on both
        tiny_model.zero_grad()
        loss = tiny_model(x, y)[1]
        loss.backward()

        shadow.forward_backward(x, y)

        # Compute gradient similarity including SNR
        metrics = shadow.compute_gradient_similarity(tiny_model)

        # Verify SNR keys present (GRAD-02 requirement)
        assert "grad_snr/mean_fp8" in metrics
        assert "grad_snr/mean_fp32" in metrics
        assert "grad_snr/mean_diff" in metrics

        # SNR should be positive (mean/std)
        assert metrics["grad_snr/mean_fp8"] > 0
        assert metrics["grad_snr/mean_fp32"] > 0

        # For identical models, diff should be ~0
        assert abs(metrics["grad_snr/mean_diff"]) < 0.1


# ============================================================================
# Trainer Tests
# ============================================================================


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def mock_data_dir(self):
        """Create temporary data directory with mock binary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal binary files
            import numpy as np

            # Need at least block_size + 1 tokens
            tokens = np.arange(100, dtype=np.uint16)  # 100 tokens

            for filename in ["train.bin", "val.bin"]:
                filepath = os.path.join(tmpdir, filename)
                arr = np.memmap(filepath, dtype=np.uint16, mode="w+", shape=(100,))
                arr[:] = tokens
                arr.flush()

            yield tmpdir

    def test_trainer_init(self, tiny_train_config, tiny_model, mock_data_dir):
        """Creates all components."""
        from altgrad.training.trainer import Trainer

        trainer = Trainer(
            config=tiny_train_config,
            model=tiny_model,
            data_dir=mock_data_dir,
            device="cpu",
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scaler is not None
        assert trainer.checkpoint_manager is not None
        assert trainer.step == 0
        assert trainer.shadow is None  # Not enabled in config

    def test_trainer_train_step(self, tiny_train_config, tiny_model, mock_data_dir):
        """Returns metrics dict with expected keys."""
        from altgrad.training.trainer import Trainer

        trainer = Trainer(
            config=tiny_train_config,
            model=tiny_model,
            data_dir=mock_data_dir,
            device="cpu",
        )

        # Get batch and run step
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))

        metrics = trainer.train_step(x, y)

        # Check expected keys
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "grad_norm" in metrics
        assert "grad_norm_l2/mean" in metrics

        # Loss should be positive
        assert metrics["loss"] > 0
        assert metrics["perplexity"] > 1

    def test_trainer_checkpoint_save_restore(
        self, tiny_train_config, tiny_model, mock_data_dir
    ):
        """Can resume from checkpoint."""
        from altgrad.training.trainer import Trainer

        with tempfile.TemporaryDirectory() as ckpt_dir:
            # Configure checkpoint directory
            tiny_train_config.checkpoint_dir = ckpt_dir
            tiny_train_config.checkpoint_interval = 2

            trainer = Trainer(
                config=tiny_train_config,
                model=tiny_model,
                data_dir=mock_data_dir,
                device="cpu",
            )

            # Run a few steps
            for _ in range(3):
                x = torch.randint(0, 100, (2, 16))
                y = torch.randint(0, 100, (2, 16))
                trainer.train_step(x, y)
                trainer.step += 1

            # Save checkpoint manually
            ckpt_path = trainer.checkpoint_manager.save(
                step=trainer.step,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scaler=trainer.scaler,
                config=tiny_train_config,
                val_loss=1.0,
            )

            # Get state before restore
            step_before = trainer.step

            # Create new trainer and resume
            new_model = GPT(GPTConfig(
                n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100
            ))
            new_trainer = Trainer(
                config=tiny_train_config,
                model=new_model,
                data_dir=mock_data_dir,
                device="cpu",
            )

            new_trainer.resume(ckpt_path)

            # Should resume from next step
            assert new_trainer.step == step_before + 1


# ============================================================================
# Integration Test
# ============================================================================


class TestIntegration:
    """Integration tests for model + shadow + trainer."""

    def test_trainer_with_shadow(self, tiny_train_config, mock_data_dir):
        """Trainer works with shadow model enabled."""
        from altgrad.training.trainer import Trainer

        # Enable shadow
        tiny_train_config.use_shadow = True

        model = GPT(GPTConfig(
            n_layer=1, n_head=1, n_embd=32, block_size=16, vocab_size=100
        ))

        trainer = Trainer(
            config=tiny_train_config,
            model=model,
            data_dir=mock_data_dir,
            device="cpu",
        )

        assert trainer.shadow is not None

        # Run step and check for shadow metrics
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))

        metrics = trainer.train_step(x, y)

        # Should have shadow metrics
        assert "shadow_loss" in metrics
        assert "grad_cos_sim/mean" in metrics
        # GRAD-02: SNR metrics
        assert "grad_snr/mean_fp8" in metrics
        assert "grad_snr/mean_fp32" in metrics
        assert "grad_snr/mean_diff" in metrics

    @pytest.fixture
    def mock_data_dir(self):
        """Create temporary data directory with mock binary files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            tokens = np.arange(100, dtype=np.uint16)

            for filename in ["train.bin", "val.bin"]:
                filepath = os.path.join(tmpdir, filename)
                arr = np.memmap(filepath, dtype=np.uint16, mode="w+", shape=(100,))
                arr[:] = tokens
                arr.flush()

            yield tmpdir
