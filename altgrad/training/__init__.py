"""Training infrastructure for AltGrad experiments.

Provides data loading, tokenization, and batch utilities for training
transformers with quantized precision. Uses nanoGPT-style binary format
for efficient memory-mapped data access.

Functions:
    prepare_eurlex: Tokenize European legal text to binary format
    get_batch: Load random batches from tokenized data
"""

from altgrad.training.data import prepare_eurlex, get_batch

__all__ = [
    "prepare_eurlex",
    "get_batch",
]
