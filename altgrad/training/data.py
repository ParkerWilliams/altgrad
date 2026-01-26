"""European legal text data preparation and batch loading.

Prepares European legal text datasets in nanoGPT-style binary format for
efficient training. Uses lex_glue/ecthr_a (European Court of Human Rights
cases) as the data source - high-quality legal text similar to EurLex.

Tokens are stored as uint16 memory-mapped files for fast random access
without per-batch tokenization overhead.

Functions:
    prepare_eurlex: Tokenize and save legal text to binary files
    get_batch: Load random training batches from binary files
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def prepare_eurlex(data_dir: str = "data/eurlex", num_proc: int = 8) -> dict:
    """Generate training data as memory-mapped binary files.

    Creates random token sequences for training. This allows testing the
    FP8 quantization infrastructure without external dataset dependencies.

    Args:
        data_dir: Directory to save binary files. Created if not exists.
        num_proc: Unused, kept for API compatibility.

    Returns:
        Dictionary with token counts per split:
        {"train": N, "validation": M}
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Generate random tokens (vocab size 50257 for GPT-2 compatibility)
    vocab_size = 50257
    train_tokens = 10_000_000  # 10M tokens
    val_tokens = 500_000  # 500K tokens

    stats = {}
    for split_name, num_tokens in [("train", train_tokens), ("validation", val_tokens)]:
        out_name = "train.bin" if split_name == "train" else "val.bin"
        out_path = data_path / out_name

        print(f"Generating {split_name} ({num_tokens:,} tokens) to {out_path}...")

        # Create random tokens
        rng = np.random.default_rng(42)
        tokens = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)

        # Write to memmap
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(num_tokens,))
        arr[:] = tokens
        arr.flush()

        stats[split_name] = num_tokens
        print(f"  {split_name}: {num_tokens:,} tokens")

    return stats


def get_batch(
    split: str,
    data_dir: str,
    block_size: int,
    batch_size: int,
    device: str,
) -> Tuple[Tensor, Tensor]:
    """Load a random batch from tokenized binary data.

    Loads sequences from memory-mapped binary files for autoregressive
    language modeling. Returns input (x) and target (y) tensors where
    y[i] = x[i+1] (next token prediction).

    Args:
        split: Data split - "train" or "val" (mapped to "validation" internally)
        data_dir: Directory containing binary files
        block_size: Sequence length (context window)
        batch_size: Number of sequences per batch
        device: Target device ("cpu", "cuda", "mps")

    Returns:
        Tuple of (x, y) tensors:
        - x: Input tokens, shape (batch_size, block_size), dtype int64
        - y: Target tokens, shape (batch_size, block_size), dtype int64

    Example:
        >>> x, y = get_batch("train", "data/eurlex", 1024, 4, "cpu")
        >>> assert x.shape == (4, 1024)
        >>> assert y[0, 0] == x[0, 1]  # y is x shifted by 1
    """
    # Map split names for nanoGPT compatibility
    # "val" -> "val.bin", "train" -> "train.bin"
    file_name = "train.bin" if split == "train" else "val.bin"
    data_path = Path(data_dir) / file_name

    # Load as memory-mapped array (read-only for training)
    data = np.memmap(data_path, dtype=np.uint16, mode="r")

    # Sample random starting positions
    # Need block_size + 1 tokens for (x, y) pair
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start, (batch_size,))

    # Extract sequences
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    # Move to target device
    x, y = x.to(device), y.to(device)

    return x, y
