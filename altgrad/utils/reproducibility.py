"""Comprehensive seed setup for reproducible ablation experiments.

Provides utilities for setting all random number generator sources
to ensure reproducible experiments where the only variable is the
quantization format.

Based on PyTorch Reproducibility Guide:
https://pytorch.org/docs/stable/notes/randomness.html

Key functions:
- set_seed_for_reproducibility: Set all RNG sources (Python, NumPy, PyTorch, CUDA)
- seed_worker: DataLoader worker seeding function
- create_reproducible_dataloader: Helper for reproducible data loading
- get_rng_state / set_rng_state: RNG state capture/restore for checkpointing

Example:
    >>> from altgrad.utils import set_seed_for_reproducibility
    >>> set_seed_for_reproducibility(42)
    >>> x1 = torch.randn(10)
    >>> set_seed_for_reproducibility(42)
    >>> x2 = torch.randn(10)
    >>> assert torch.allclose(x1, x2)  # Identical!
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed_for_reproducibility(seed: int) -> None:
    """Set all random seeds for reproducible experiments.

    Sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch's random generator (CPU and CUDA)
    - cuDNN deterministic mode
    - CUBLAS workspace config (for CUDA >= 10.2)

    This function should be called at the start of each ablation run
    to ensure identical initial conditions. The only difference between
    runs should be the quantization format.

    Args:
        seed: Random seed to use (e.g., 42)

    Example:
        >>> # Ablation experiment setup
        >>> set_seed_for_reproducibility(42)
        >>> model1 = GPT(config)
        >>> quantize_model(model1, E5M2)
        >>>
        >>> set_seed_for_reproducibility(42)
        >>> model2 = GPT(config)
        >>> quantize_model(model2, E3M4)
        >>> # model1 and model2 have identical initial weights
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode
    # Note: may reduce performance but ensures reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS workspace config (for reproducibility on CUDA >= 10.2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # PyTorch 1.8+ use_deterministic_algorithms
    # Some operations may not have deterministic implementations
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            # (e.g., scatter_add on CUDA)
            pass


def seed_worker(worker_id: int) -> None:
    """Seed function for DataLoader workers.

    Ensures each worker has reproducible random state based on
    the initial seed and worker ID. Use this with DataLoader's
    worker_init_fn parameter.

    Args:
        worker_id: DataLoader worker ID (0 to num_workers-1)

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker,
        ... )
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs: Any,
) -> DataLoader:
    """Create a DataLoader with reproducible shuffling and worker seeding.

    Configures a DataLoader for reproducible behavior:
    - Uses a seeded Generator for shuffling
    - Seeds workers with seed_worker function
    - Same seed produces identical batch ordering across runs

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        seed: Random seed for shuffling
        shuffle: Whether to shuffle (default True)
        num_workers: Number of worker processes (default 0)
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader with reproducible behavior

    Example:
        >>> loader = create_reproducible_dataloader(dataset, 32, seed=42)
        >>> batches1 = [b for b in loader]
        >>> loader = create_reproducible_dataloader(dataset, 32, seed=42)
        >>> batches2 = [b for b in loader]
        >>> # batches1 and batches2 are identical
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator if shuffle else None,
        **kwargs,
    )


def get_rng_state() -> Dict[str, Any]:
    """Capture current RNG state for all sources.

    Captures state for Python, NumPy, and PyTorch RNGs. Useful for
    checkpointing during training to enable exact restoration.

    Returns:
        Dictionary with RNG states that can be restored with set_rng_state()

    Example:
        >>> # During training
        >>> state = get_rng_state()
        >>> save_checkpoint(model, optimizer, rng_state=state)
        >>>
        >>> # When resuming
        >>> checkpoint = load_checkpoint(path)
        >>> set_rng_state(checkpoint['rng_state'])
    """
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from captured state.

    Restores Python, NumPy, and PyTorch RNG states from a dictionary
    created by get_rng_state(). Enables exact restoration of random
    sequence for checkpoint resumption.

    Args:
        state: State dictionary from get_rng_state()

    Example:
        >>> state = get_rng_state()
        >>> x1 = torch.randn(10)
        >>> set_rng_state(state)
        >>> x2 = torch.randn(10)
        >>> assert torch.allclose(x1, x2)
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


__all__ = [
    "set_seed_for_reproducibility",
    "seed_worker",
    "create_reproducible_dataloader",
    "get_rng_state",
    "set_rng_state",
]
