"""European legal text data preparation and batch loading.

Prepares EUR-Lex legal text dataset in nanoGPT-style binary format for
efficient training.

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
import tiktoken
import torch
from datasets import load_dataset
from torch import Tensor


def prepare_eurlex(data_dir: str = "data/eurlex", num_proc: int = 8) -> dict:
    """Tokenize EUR-Lex legal text and save as memory-mapped binary files.

    Args:
        data_dir: Directory to save binary files. Created if not exists.
        num_proc: Number of processes for parallel tokenization.

    Returns:
        Dictionary with token counts per split:
        {"train": N, "validation": M}
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Load EUR-Lex dataset
    print("Loading EUR-Lex dataset...")
    dataset = load_dataset("NLP-AUEB/eurlex", split="train", trust_remote_code=True)

    # Initialize GPT-2 tokenizer (50257 vocab)
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token

    def tokenize(example):
        """Tokenize text and append EOT token."""
        text = example["text"]
        if not text or not text.strip():
            return {"ids": [], "len": 0}
        ids = enc.encode_ordinary(text)
        ids.append(eot_token)
        return {"ids": ids, "len": len(ids)}

    # Tokenize
    print(f"Tokenizing with {num_proc} processes...")
    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing",
    )
    tokenized = tokenized.filter(lambda x: x["len"] > 0)

    # Split into train/val (90/10)
    tokenized = tokenized.train_test_split(test_size=0.1, seed=42)

    stats = {}
    for split_name, split_key in [("train", "train"), ("validation", "test")]:
        dset = tokenized[split_key]
        total_tokens = sum(dset["len"])

        out_name = "train.bin" if split_name == "train" else "val.bin"
        out_path = data_path / out_name

        print(f"Writing {split_name} ({total_tokens:,} tokens) to {out_path}...")

        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

        idx = 0
        for example in dset:
            ids = np.array(example["ids"], dtype=np.uint16)
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)

        arr.flush()
        stats[split_name] = total_tokens
        print(f"  {split_name}: {total_tokens:,} tokens")

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
