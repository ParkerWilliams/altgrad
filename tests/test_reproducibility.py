"""Tests for reproducibility utilities.

Tests validate that seed setup produces identical random sequences
across Python, NumPy, and PyTorch RNGs, enabling reproducible ablation
experiments where the only variable is the quantization format.
"""

import random

import numpy as np
import pytest
import torch

from altgrad.training.model import GPT, GPTConfig


class TestSetSeedTorch:
    """Test set_seed_for_reproducibility() with PyTorch RNG."""

    def test_set_seed_produces_same_torch_randn(self):
        """Same seed should produce identical torch.randn sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = torch.randn(100)

        set_seed_for_reproducibility(42)
        x2 = torch.randn(100)

        assert torch.allclose(x1, x2), "Same seed should produce identical torch.randn"

    def test_set_seed_produces_same_torch_rand(self):
        """Same seed should produce identical torch.rand sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = torch.rand(100)

        set_seed_for_reproducibility(42)
        x2 = torch.rand(100)

        assert torch.allclose(x1, x2), "Same seed should produce identical torch.rand"


class TestSetSeedNumpy:
    """Test set_seed_for_reproducibility() with NumPy RNG."""

    def test_set_seed_produces_same_numpy_randn(self):
        """Same seed should produce identical np.random.randn sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = np.random.randn(100)

        set_seed_for_reproducibility(42)
        x2 = np.random.randn(100)

        assert np.allclose(x1, x2), "Same seed should produce identical numpy.randn"

    def test_set_seed_produces_same_numpy_rand(self):
        """Same seed should produce identical np.random.rand sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = np.random.rand(100)

        set_seed_for_reproducibility(42)
        x2 = np.random.rand(100)

        assert np.allclose(x1, x2), "Same seed should produce identical numpy.rand"


class TestSetSeedPython:
    """Test set_seed_for_reproducibility() with Python random module."""

    def test_set_seed_produces_same_python_random(self):
        """Same seed should produce identical random.random sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = [random.random() for _ in range(100)]

        set_seed_for_reproducibility(42)
        x2 = [random.random() for _ in range(100)]

        assert x1 == x2, "Same seed should produce identical random.random"

    def test_set_seed_produces_same_python_randint(self):
        """Same seed should produce identical random.randint sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = [random.randint(0, 1000) for _ in range(100)]

        set_seed_for_reproducibility(42)
        x2 = [random.randint(0, 1000) for _ in range(100)]

        assert x1 == x2, "Same seed should produce identical random.randint"


class TestDifferentSeeds:
    """Test that different seeds produce different results."""

    def test_different_seeds_produce_different_torch(self):
        """Different seeds should produce different torch.randn sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = torch.randn(100)

        set_seed_for_reproducibility(43)
        x2 = torch.randn(100)

        assert not torch.allclose(x1, x2), "Different seeds should produce different results"

    def test_different_seeds_produce_different_numpy(self):
        """Different seeds should produce different np.random.randn sequences."""
        from altgrad.utils import set_seed_for_reproducibility

        set_seed_for_reproducibility(42)
        x1 = np.random.randn(100)

        set_seed_for_reproducibility(43)
        x2 = np.random.randn(100)

        assert not np.allclose(x1, x2), "Different seeds should produce different results"


class TestGPTModelReproducibility:
    """Test GPT model initialization reproducibility."""

    def test_gpt_model_init_reproducible(self):
        """Same seed should produce identical GPT model weights."""
        from altgrad.utils import set_seed_for_reproducibility

        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)

        set_seed_for_reproducibility(42)
        model1 = GPT(config)
        weights1 = model1.transformer.h[0].mlp.c_fc.weight.clone()

        set_seed_for_reproducibility(42)
        model2 = GPT(config)
        weights2 = model2.transformer.h[0].mlp.c_fc.weight.clone()

        assert torch.allclose(weights1, weights2), "Same seed should produce identical model weights"

    def test_gpt_model_all_weights_reproducible(self):
        """All GPT weights should be identical with same seed."""
        from altgrad.utils import set_seed_for_reproducibility

        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)

        set_seed_for_reproducibility(42)
        model1 = GPT(config)

        set_seed_for_reproducibility(42)
        model2 = GPT(config)

        # Check all parameters
        for (name1, p1), (name2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2, f"Parameter names should match: {name1} vs {name2}"
            assert torch.allclose(p1, p2), f"Parameter {name1} should be identical"


class TestAblationReproducibility:
    """Test ablation experiment reproducibility."""

    def test_ablation_same_seed_identical_initial_state(self):
        """Ablation runs with same seed should have identical initial weights."""
        from altgrad.utils import set_seed_for_reproducibility

        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)

        # Run 1: Setup for E5M2 experiment
        set_seed_for_reproducibility(42)
        model1 = GPT(config)
        weights1_before = model1.transformer.h[0].mlp.c_fc.weight.clone()

        # Run 2: Setup for E3M4 experiment (same seed)
        set_seed_for_reproducibility(42)
        model2 = GPT(config)
        weights2_before = model2.transformer.h[0].mlp.c_fc.weight.clone()

        # Initial weights should be identical before quantization
        assert torch.allclose(weights1_before, weights2_before), \
            "Initial weights should match with same seed"

    def test_ablation_same_seed_different_format_different_output(self):
        """Same seed, different formats should produce different outputs (after quantization)."""
        from altgrad.integration import quantize_model
        from altgrad.quantization import E3M4, E5M2
        from altgrad.utils import set_seed_for_reproducibility

        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)

        # Run 1: E5M2 quantization
        set_seed_for_reproducibility(42)
        model1 = GPT(config)
        quantize_model(model1, E5M2, skip_patterns=["lm_head"])

        # Run 2: E3M4 quantization (same seed)
        set_seed_for_reproducibility(42)
        model2 = GPT(config)
        quantize_model(model2, E3M4, skip_patterns=["lm_head"])

        # Same input
        set_seed_for_reproducibility(123)
        x = torch.randint(0, 100, (2, 10))

        # Forward passes should produce different results due to quantization
        model1.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        # Outputs should differ because E5M2 and E3M4 quantize differently
        assert not torch.allclose(out1, out2, atol=1e-3), \
            "Different quantization formats should produce different outputs"


class TestSeedWorker:
    """Test seed_worker function for DataLoader."""

    def test_seed_worker_exists(self):
        """seed_worker should be importable."""
        from altgrad.utils import seed_worker

        # Should be callable
        assert callable(seed_worker)


class TestCreateReproducibleDataloader:
    """Test create_reproducible_dataloader helper."""

    def test_create_reproducible_dataloader_exists(self):
        """create_reproducible_dataloader should be importable."""
        from altgrad.utils import create_reproducible_dataloader

        # Should be callable
        assert callable(create_reproducible_dataloader)

    def test_create_reproducible_dataloader_produces_same_batches(self):
        """Same seed should produce identical batch ordering."""
        from torch.utils.data import TensorDataset

        from altgrad.utils import create_reproducible_dataloader

        # Create simple dataset
        data = torch.arange(100).float()
        dataset = TensorDataset(data)

        # Create two loaders with same seed
        loader1 = create_reproducible_dataloader(dataset, batch_size=10, seed=42)
        loader2 = create_reproducible_dataloader(dataset, batch_size=10, seed=42)

        # Batches should be identical
        for batch1, batch2 in zip(loader1, loader2):
            assert torch.equal(batch1[0], batch2[0]), "Batches should be identical with same seed"


class TestRNGState:
    """Test get_rng_state and set_rng_state for checkpointing."""

    def test_get_set_rng_state_roundtrip(self):
        """get_rng_state/set_rng_state should enable exact RNG restoration."""
        from altgrad.utils import get_rng_state, set_rng_state, set_seed_for_reproducibility

        set_seed_for_reproducibility(42)

        # Generate some random numbers
        _ = torch.randn(10)
        _ = np.random.randn(10)
        _ = random.random()

        # Save state
        state = get_rng_state()

        # Generate more random numbers
        x1_torch = torch.randn(10)
        x1_numpy = np.random.randn(10)
        x1_python = random.random()

        # Restore state
        set_rng_state(state)

        # Should get same numbers
        x2_torch = torch.randn(10)
        x2_numpy = np.random.randn(10)
        x2_python = random.random()

        assert torch.allclose(x1_torch, x2_torch), "Torch RNG should be restored"
        assert np.allclose(x1_numpy, x2_numpy), "NumPy RNG should be restored"
        assert x1_python == x2_python, "Python RNG should be restored"
