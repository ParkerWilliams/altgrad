"""Checkpoint save/load utilities with full training state.

Provides functions and a manager class for saving and restoring training
state, including model weights, optimizer state, scaler state, quantization
state, and RNG state for reproducibility.

Key features:
  - Full RNG state restoration (random, numpy, torch, cuda)
  - Quantization state preservation (amax_history, scale_factors)
  - Automatic checkpoint rotation (keep N most recent + best)
  - Anomaly checkpoints for debugging failures

Example:
    >>> from altgrad.training.checkpoint import save_checkpoint, load_checkpoint
    >>> save_checkpoint('ckpt.pt', model, optimizer, scaler, step=100, config=cfg)
    >>> step, cfg, quant = load_checkpoint('ckpt.pt', model, optimizer, scaler)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Optional[Any],
    step: int,
    config: Any,
    quantization_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save complete training checkpoint.

    Saves all state needed to resume training exactly:
      - Model state dict
      - Optimizer state dict
      - Scaler state dict (if provided)
      - Training step
      - Configuration (as dict)
      - Quantization state (amax_history, scale_factors)
      - RNG state (random, numpy, torch, cuda)

    Args:
        filepath: Path to save checkpoint
        model: PyTorch model
        optimizer: Optimizer instance
        scaler: Optional GradScaler for mixed precision
        step: Current training step
        config: Configuration (dataclass with asdict or dict)
        quantization_state: Optional dict with quantization info

    Example:
        >>> save_checkpoint(
        ...     'step_100.pt',
        ...     model, optimizer, scaler,
        ...     step=100, config=train_config,
        ...     quantization_state={'amax_history': [...]}
        ... )
    """
    # Convert config to dict if it's a dataclass
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {"raw": str(config)}

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config_dict,
        # RNG state
        "rng_state": {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }

    # Optional scaler state
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # CUDA RNG state (all devices)
    if torch.cuda.is_available():
        checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

    # Quantization state
    if quantization_state is not None:
        checkpoint["quantization_state"] = quantization_state

    # Ensure directory exists
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Optional[Any] = None,
) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
    """Load checkpoint and restore training state.

    Restores all state from checkpoint:
      - Model weights
      - Optimizer state
      - Scaler state (if scaler provided)
      - RNG state (for reproducibility)

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optimizer to restore state
        scaler: Optional GradScaler to restore

    Returns:
        Tuple of:
          - step: Training step when checkpoint was saved
          - config: Configuration dict
          - quantization_state: Quantization state dict (or None)

    Raises:
        FileNotFoundError: If checkpoint doesn't exist

    Example:
        >>> step, config, quant_state = load_checkpoint(
        ...     'step_100.pt', model, optimizer, scaler
        ... )
        >>> print(f"Resuming from step {step}")
    """
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    # Restore model and optimizer
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scaler if provided and saved
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore RNG state
    rng_state = checkpoint.get("rng_state", {})
    if "random" in rng_state:
        random.setstate(rng_state["random"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if "cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])

    step = checkpoint.get("step", 0)
    config = checkpoint.get("config", {})
    quantization_state = checkpoint.get("quantization_state")

    return step, config, quantization_state


class CheckpointManager:
    """Manages checkpoint saving with automatic rotation.

    Keeps the N most recent checkpoints plus the best checkpoint
    (lowest validation loss). Supports anomaly checkpoints for
    debugging training failures.

    Attributes:
        checkpoint_dir: Directory for checkpoints
        max_checkpoints: Maximum regular checkpoints to keep
        checkpoints: List of (step, filepath) tuples
        best_loss: Best validation loss seen
        best_path: Path to best checkpoint

    Example:
        >>> manager = CheckpointManager('checkpoints', max_checkpoints=3)
        >>> for step in range(1000):
        ...     # training...
        ...     if step % 100 == 0:
        ...         manager.save(step, model, opt, scaler, cfg, val_loss)
        >>> print(f"Best checkpoint: {manager.best()}")
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 3):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum checkpoints to retain (plus best)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

        # Track saved checkpoints: (step, filepath)
        self.checkpoints: List[Tuple[int, str]] = []

        # Best checkpoint tracking
        self.best_loss: Optional[float] = None
        self.best_path: Optional[str] = None

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scaler: Optional[Any],
        config: Any,
        val_loss: float,
        quantization_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint and manage rotation.

        Args:
            step: Current training step
            model: PyTorch model
            optimizer: Optimizer instance
            scaler: Optional GradScaler
            config: Configuration
            val_loss: Validation loss for best tracking
            quantization_state: Optional quantization state

        Returns:
            Path to saved checkpoint
        """
        filepath = str(self.checkpoint_dir / f"step_{step}.pt")

        save_checkpoint(
            filepath, model, optimizer, scaler, step, config, quantization_state
        )

        self.checkpoints.append((step, filepath))

        # Update best if improved
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = str(self.checkpoint_dir / "best.pt")
            save_checkpoint(
                best_path, model, optimizer, scaler, step, config, quantization_state
            )
            self.best_path = best_path

        # Rotate old checkpoints (keep max_checkpoints most recent)
        while len(self.checkpoints) > self.max_checkpoints:
            _, old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        return filepath

    def save_on_anomaly(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scaler: Optional[Any],
        config: Any,
        quantization_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save emergency checkpoint on anomaly detection.

        These are not subject to rotation and are kept for debugging.

        Args:
            step: Current training step
            model: PyTorch model
            optimizer: Optimizer instance
            scaler: Optional GradScaler
            config: Configuration
            quantization_state: Optional quantization state

        Returns:
            Path to anomaly checkpoint
        """
        filepath = str(self.checkpoint_dir / f"anomaly_{step}.pt")
        save_checkpoint(
            filepath, model, optimizer, scaler, step, config, quantization_state
        )
        return filepath

    def latest(self) -> Optional[str]:
        """Get path to most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1][1]

    def best(self) -> Optional[str]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint, or None if no checkpoints
        """
        return self.best_path


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
]
