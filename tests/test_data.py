"""Tests for data loading and batch utilities.

Tests validate:
1. Binary file existence and format (uint16)
2. get_batch() tensor shapes and dtypes
3. Autoregressive target shift (y = x shifted by 1)
4. Device placement
5. Reproducibility with seeded random state
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from altgrad.training.data import get_batch


# Data directory path
DATA_DIR = Path(__file__).parent.parent / "data" / "eurlex"


def data_exists() -> bool:
    """Check if prepared data files exist."""
    train_path = DATA_DIR / "train.bin"
    val_path = DATA_DIR / "val.bin"
    return train_path.exists() and val_path.exists()


# Skip all tests if data doesn't exist
pytestmark = pytest.mark.skipif(
    not data_exists(),
    reason="Data files not found. Run prepare_eurlex() first.",
)


class TestBinaryFiles:
    """Test binary file existence and format."""

    def test_binary_files_exist(self):
        """Verify train.bin and val.bin exist after prepare_eurlex()."""
        train_path = DATA_DIR / "train.bin"
        val_path = DATA_DIR / "val.bin"
        assert train_path.exists(), f"train.bin not found at {train_path}"
        assert val_path.exists(), f"val.bin not found at {val_path}"

    def test_binary_file_dtype(self):
        """Verify files are stored as uint16."""
        train_path = DATA_DIR / "train.bin"
        val_path = DATA_DIR / "val.bin"

        # Load as memmap and check dtype
        train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
        val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

        assert train_data.dtype == np.uint16, f"train.bin dtype is {train_data.dtype}"
        assert val_data.dtype == np.uint16, f"val.bin dtype is {val_data.dtype}"

    def test_binary_files_non_empty(self):
        """Verify binary files contain tokens."""
        train_path = DATA_DIR / "train.bin"
        val_path = DATA_DIR / "val.bin"

        train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
        val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

        assert len(train_data) > 0, "train.bin is empty"
        assert len(val_data) > 0, "val.bin is empty"

    def test_token_values_in_vocab_range(self):
        """Verify all tokens are within GPT-2 vocab range (0-50256)."""
        train_path = DATA_DIR / "train.bin"
        train_data = np.memmap(train_path, dtype=np.uint16, mode="r")

        # GPT-2 vocab size is 50257 (indices 0-50256)
        assert train_data.min() >= 0, f"Token below 0: {train_data.min()}"
        assert train_data.max() <= 50256, f"Token above vocab: {train_data.max()}"


class TestGetBatchShapes:
    """Test get_batch() return tensor shapes."""

    def test_get_batch_shapes(self):
        """get_batch returns (batch_size, block_size) tensors."""
        block_size = 64
        batch_size = 4

        x, y = get_batch("train", str(DATA_DIR), block_size, batch_size, "cpu")

        assert x.shape == (batch_size, block_size), f"x shape: {x.shape}"
        assert y.shape == (batch_size, block_size), f"y shape: {y.shape}"

    def test_get_batch_val_split(self):
        """get_batch works for val split."""
        block_size = 64
        batch_size = 4

        x, y = get_batch("val", str(DATA_DIR), block_size, batch_size, "cpu")

        assert x.shape == (batch_size, block_size)
        assert y.shape == (batch_size, block_size)

    def test_get_batch_dtype(self):
        """get_batch returns int64 tensors for indexing."""
        x, y = get_batch("train", str(DATA_DIR), 64, 4, "cpu")

        assert x.dtype == torch.int64, f"x dtype: {x.dtype}"
        assert y.dtype == torch.int64, f"y dtype: {y.dtype}"


class TestGetBatchTargetShift:
    """Test autoregressive target shift: y[i] = x[i+1]."""

    def test_get_batch_target_shift(self):
        """y[i] == x[i+1] for autoregressive language modeling."""
        block_size = 64
        batch_size = 4

        x, y = get_batch("train", str(DATA_DIR), block_size, batch_size, "cpu")

        # For each sequence in batch, y should be x shifted by 1
        # This means: y[b, i] should equal x[b, i+1] for most positions
        # But we need to verify using the raw data

        # Load raw data to verify
        data = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")

        # The relationship should be: for a starting position p,
        # x[b] = data[p:p+block_size]
        # y[b] = data[p+1:p+1+block_size]
        # Therefore: y[b, 0] == x[b, 1], y[b, 1] == x[b, 2], etc.

        # Verify for all sequences
        for b in range(batch_size):
            # Check that y[b, :-1] == x[b, 1:]
            # This is the same as: y[b, i] == x[b, i+1] for i in [0, block_size-2]
            assert torch.all(y[b, :-1] == x[b, 1:]), f"Target shift mismatch in batch {b}"

    def test_get_batch_consecutive_tokens(self):
        """Verify x and y represent consecutive token windows."""
        block_size = 64
        batch_size = 2

        x, y = get_batch("train", str(DATA_DIR), block_size, batch_size, "cpu")

        # The first token of y should equal the second token of x
        # (they're both the token at position p+1 in the data)
        assert torch.all(y[:, 0] == x[:, 1]), "y[:,0] should equal x[:,1]"


class TestGetBatchDevice:
    """Test device placement."""

    def test_get_batch_device_cpu(self):
        """Tensors are on correct device (cpu)."""
        x, y = get_batch("train", str(DATA_DIR), 64, 4, "cpu")

        assert x.device.type == "cpu", f"x device: {x.device}"
        assert y.device.type == "cpu", f"y device: {y.device}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_batch_device_cuda(self):
        """Tensors are on CUDA when requested."""
        x, y = get_batch("train", str(DATA_DIR), 64, 4, "cuda")

        assert x.device.type == "cuda", f"x device: {x.device}"
        assert y.device.type == "cuda", f"y device: {y.device}"

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="MPS not available",
    )
    def test_get_batch_device_mps(self):
        """Tensors are on MPS when requested."""
        x, y = get_batch("train", str(DATA_DIR), 64, 4, "mps")

        assert x.device.type == "mps", f"x device: {x.device}"
        assert y.device.type == "mps", f"y device: {y.device}"


class TestGetBatchReproducibility:
    """Test reproducibility with seeded random state."""

    def test_get_batch_reproducibility(self):
        """Same random state gives same batch."""
        block_size = 64
        batch_size = 4

        # Set seed and get first batch
        torch.manual_seed(42)
        x1, y1 = get_batch("train", str(DATA_DIR), block_size, batch_size, "cpu")

        # Set same seed and get second batch
        torch.manual_seed(42)
        x2, y2 = get_batch("train", str(DATA_DIR), block_size, batch_size, "cpu")

        assert torch.all(x1 == x2), "x tensors should be identical with same seed"
        assert torch.all(y1 == y2), "y tensors should be identical with same seed"

    def test_get_batch_different_seeds(self):
        """Different seeds give different batches."""
        block_size = 64
        batch_size = 4

        torch.manual_seed(42)
        x1, _ = get_batch("train", str(DATA_DIR), block_size, batch_size, "cpu")

        torch.manual_seed(123)
        x2, _ = get_batch("train", str(DATA_DIR), block_size, batch_size, "cpu")

        # Batches should be different (extremely unlikely to be same)
        assert not torch.all(x1 == x2), "Different seeds should give different batches"
