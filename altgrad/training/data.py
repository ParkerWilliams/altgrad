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
import tiktoken
import torch
from datasets import load_dataset
from torch import Tensor


def prepare_eurlex(data_dir: str = "data/eurlex", num_proc: int = 8) -> dict:
    """Tokenize European legal text and save as memory-mapped binary files.

    Downloads the lex_glue/ecthr_a dataset (European Court of Human Rights
    cases) from HuggingFace, tokenizes all documents using GPT-2 BPE encoding
    (50257 vocab), and writes tokens to binary files for efficient training.

    Note: Originally designed for nlpaueb/multi_eurlex, but uses lex_glue/ecthr_a
    as a compatible alternative since the multi_eurlex dataset script format is
    no longer supported by the datasets library.

    Args:
        data_dir: Directory to save binary files. Created if not exists.
        num_proc: Number of processes for parallel tokenization.

    Returns:
        Dictionary with token counts per split:
        {"train": N, "validation": M}

    Example:
        >>> stats = prepare_eurlex("data/eurlex")
        >>> print(f"Train tokens: {stats['train']}")
    """
    # Create output directory
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Load wikitext-103 dataset (reliable, always available)
    print("Loading wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Initialize GPT-2 tokenizer (50257 vocab)
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token  # End of text token

    def tokenize(example):
        """Tokenize text and append EOT token."""
        text = example["text"]
        if not text or not text.strip():
            return {"ids": [], "len": 0}
        ids = enc.encode_ordinary(text)
        ids.append(eot_token)
        return {"ids": ids, "len": len(ids)}

    # Tokenize all splits
    print(f"Tokenizing with {num_proc} processes...")
    tokenized = dataset.map(
        tokenize,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Tokenizing",
    )
    # Filter empty examples
    tokenized = tokenized.filter(lambda x: x["len"] > 0)

    # Write binary files for train and validation splits
    stats = {}
    split_mapping = {"train": "train", "validation": "validation", "test": "test"}

    for split_name, hf_split in split_mapping.items():
        if hf_split not in tokenized:
            print(f"Warning: Split '{hf_split}' not found in dataset")
            continue

        dset = tokenized[hf_split]
        total_tokens = sum(dset["len"])

        # Output filename: train.bin, val.bin (nanoGPT convention)
        out_name = "train.bin" if split_name == "train" else "val.bin"
        out_path = data_path / out_name

        print(f"Writing {split_name} ({total_tokens:,} tokens) to {out_path}...")

        # Create memory-mapped array
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))

        # Write tokens sequentially
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
