"""Utility functions for AltGrad experiments.

This module provides utilities for reproducibility, logging, and other
experiment management tasks.

Key functions:
- set_seed_for_reproducibility: Set all RNG sources for reproducible runs
- seed_worker: DataLoader worker seeding function
- create_reproducible_dataloader: Helper for reproducible data loading
- get_rng_state / set_rng_state: RNG state capture/restore

Example:
    >>> from altgrad.utils import set_seed_for_reproducibility
    >>> set_seed_for_reproducibility(42)
    >>> # All random operations now deterministic
"""

from altgrad.utils.reproducibility import (
    set_seed_for_reproducibility,
    seed_worker,
    create_reproducible_dataloader,
    get_rng_state,
    set_rng_state,
)

__all__ = [
    "set_seed_for_reproducibility",
    "seed_worker",
    "create_reproducible_dataloader",
    "get_rng_state",
    "set_rng_state",
]
