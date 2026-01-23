#!/usr/bin/env python
"""Run AltGrad training experiment.

This script provides the entry point for running training experiments.
It loads a YAML configuration, creates the model and trainer, and
runs the training loop.

Example:
    python experiments/run_experiment.py experiments/configs/bf16_baseline.yaml
    python experiments/run_experiment.py experiments/configs/bf16_baseline.yaml --device cuda
    python experiments/run_experiment.py experiments/configs/bf16_baseline.yaml --resume checkpoints/bf16_baseline/step_100.pt
"""
import argparse
import os
import random
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from altgrad.training import TrainConfig, load_config, GPT, GPTConfig, Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run AltGrad experiment")
    parser.add_argument("config", help="Path to experiment config YAML")
    parser.add_argument("--data-dir", default="data/eurlex", help="Data directory")
    parser.add_argument("--resume", help="Checkpoint to resume from")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu/mps)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {config.run_name}")
    print(f"  Model: {config.n_layer} layers, {config.n_embd} dim, {config.n_head} heads")
    print(f"  Training: {config.max_steps} steps, batch_size={config.batch_size}")
    print(f"  Device: {args.device}")

    # Set seed for reproducibility
    set_seed(config.seed)
    print(f"  Seed: {config.seed}")

    # Create model
    gpt_config = GPTConfig(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
    )
    model = GPT(gpt_config).to(args.device)

    # Report model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Create trainer
    trainer = Trainer(config, model, args.data_dir, args.device)

    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Train
    print("\nStarting training...")
    trainer.train()
    print("\nTraining complete!")
    print("\n" + "=" * 60)
    print("NEXT STEP: Regenerate analysis reports")
    print("=" * 60)
    print("Run: python scripts/generate_reports.py --project <your-wandb-project>")
    print("This will update ANAL-01, ANAL-02, ANAL-03 with your new results.")


if __name__ == "__main__":
    main()
