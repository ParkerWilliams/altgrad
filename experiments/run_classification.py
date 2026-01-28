#!/usr/bin/env python3
"""Run classification experiments with full metrics.

This is the correct experiment runner for the research objective:
Determine which FP8 datatype benefits most from manifold-aware gradient updates.

Measures:
1. Throughput (samples/sec)
2. Classification performance (F1 micro, F1 macro)
3. Rank stability (stable rank, effective rank, collapse detection)
4. Training efficiency (weight updates, stall ratio, flips)

Experimental Matrix: 6 formats x 2 optimizers = 12 conditions
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from altgrad.training.classifier import TransformerClassifier, ClassifierConfig
from altgrad.training.classification_data import create_dataloaders
from altgrad.training.classification_trainer import (
    ClassificationTrainer,
    ClassificationTrainConfig,
)


# Experiment matrix: 6 FP8 formats x 2 optimizers
FORMATS = ["BF16", "E5M2", "E3M4", "E1M6", "E0M7", "E7M0"]
OPTIMIZERS = ["adamw", "manifold"]


def run_experiment(
    fp8_format: str,
    optimizer: str,
    config_overrides: dict = None,
) -> dict:
    """Run a single experiment condition.

    Args:
        fp8_format: FP8 format name (BF16, E5M2, E3M4, E1M6, E0M7, E7M0)
        optimizer: Optimizer type (adamw, manifold)
        config_overrides: Optional config overrides

    Returns:
        Dictionary with experiment results
    """
    config_overrides = config_overrides or {}

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Running: {fp8_format} + {optimizer}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Data
    max_examples = config_overrides.get("max_examples", None)
    batch_size = config_overrides.get("batch_size", 16)
    max_length = config_overrides.get("max_length", 512)

    print("Loading EUR-Lex dataset...")
    train_loader, val_loader, test_loader, num_labels = create_dataloaders(
        batch_size=batch_size,
        max_length=max_length,
        max_examples=max_examples,
    )
    print(f"Loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Number of labels: {num_labels}")

    # Model
    pos_weight = config_overrides.get("pos_weight", 10.0)  # Default: 10x penalty for FN
    model_config = ClassifierConfig(
        num_labels=num_labels,
        quantize_encoder=(fp8_format != "BF16"),
        quantize_classifier=False,
        pos_weight=pos_weight,
    )
    model = TransformerClassifier(model_config)
    print(f"Positive class weight: {pos_weight}")

    # Training config
    use_fp8 = (fp8_format != "BF16")
    use_manifold = (optimizer == "manifold")

    train_config = ClassificationTrainConfig(
        learning_rate=config_overrides.get("learning_rate", 2e-5),
        weight_decay=config_overrides.get("weight_decay", 0.01),
        max_steps=config_overrides.get("max_steps", 5000),
        batch_size=batch_size,
        max_length=max_length,
        grad_clip=config_overrides.get("grad_clip", 1.0),
        eval_interval=config_overrides.get("eval_interval", 500),
        log_interval=config_overrides.get("log_interval", 50),
        use_fp8=use_fp8,
        fp8_format=fp8_format if use_fp8 else "E5M2",
        use_manifold_aware=use_manifold,
        rank_log_interval=config_overrides.get("rank_log_interval", 100),
        rank_warn_threshold=config_overrides.get("rank_warn_threshold", 0.3),
        track_flips=use_fp8,
        checkpoint_dir=config_overrides.get("checkpoint_dir", f"checkpoints/{fp8_format}_{optimizer}"),
        checkpoint_interval=config_overrides.get("checkpoint_interval", 1000),
        project=config_overrides.get("wandb_project", "altgrad-classification"),
        run_name=f"{fp8_format}_{optimizer}",
    )

    # Train
    trainer = ClassificationTrainer(
        config=train_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    trainer.train()

    # Final evaluation
    final_metrics = trainer.evaluate()

    results = {
        "format": fp8_format,
        "optimizer": optimizer,
        "f1_micro": final_metrics.f1_micro,
        "f1_macro": final_metrics.f1_macro,
        "precision_micro": final_metrics.precision_micro,
        "recall_micro": final_metrics.recall_micro,
        "hamming_loss": final_metrics.hamming_loss,
        "roc_auc_micro": final_metrics.roc_auc_micro,
        "best_f1_micro": trainer.best_f1_micro,
    }

    print(f"\n{'='*60}")
    print(f"Results: {fp8_format} + {optimizer}")
    print(f"  F1 micro: {results['f1_micro']:.4f}")
    print(f"  F1 macro: {results['f1_macro']:.4f}")
    print(f"  Best F1:  {results['best_f1_micro']:.4f}")
    print(f"{'='*60}\n")

    return results


def run_full_matrix(config_overrides: dict = None) -> list:
    """Run all 12 experiment conditions.

    Args:
        config_overrides: Optional config overrides

    Returns:
        List of result dictionaries
    """
    all_results = []

    for fp8_format in FORMATS:
        for optimizer in OPTIMIZERS:
            try:
                results = run_experiment(fp8_format, optimizer, config_overrides)
                all_results.append(results)
            except Exception as e:
                print(f"FAILED: {fp8_format} + {optimizer}: {e}")
                all_results.append({
                    "format": fp8_format,
                    "optimizer": optimizer,
                    "error": str(e),
                })

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run classification experiments")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--format", type=str, choices=FORMATS, help="Single format to run")
    parser.add_argument("--optimizer", type=str, choices=OPTIMIZERS, help="Single optimizer to run")
    parser.add_argument("--max-examples", type=int, help="Limit examples per split")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project (disabled if not set)")
    parser.add_argument("--pos-weight", type=float, default=10.0, help="Positive class weight (penalize FN)")
    parser.add_argument("--full-matrix", action="store_true", help="Run all 12 conditions")

    args = parser.parse_args()

    # Load config from file if provided
    config_overrides = {}
    if args.config:
        with open(args.config) as f:
            config_overrides = yaml.safe_load(f)

    # Override with command line args
    if args.max_examples:
        config_overrides["max_examples"] = args.max_examples
    if args.max_steps:
        config_overrides["max_steps"] = args.max_steps
    if args.batch_size:
        config_overrides["batch_size"] = args.batch_size
    if args.wandb_project is not None:
        config_overrides["wandb_project"] = args.wandb_project
    if args.pos_weight:
        config_overrides["pos_weight"] = args.pos_weight

    # Run experiments
    if args.full_matrix:
        results = run_full_matrix(config_overrides)
        print("\n" + "="*60)
        print("FULL MATRIX RESULTS")
        print("="*60)
        for r in results:
            if "error" in r:
                print(f"{r['format']:6} + {r['optimizer']:8}: FAILED - {r['error']}")
            else:
                print(f"{r['format']:6} + {r['optimizer']:8}: F1={r['f1_micro']:.4f}")

    elif args.format and args.optimizer:
        run_experiment(args.format, args.optimizer, config_overrides)

    elif args.format:
        # Run both optimizers for this format
        for optimizer in OPTIMIZERS:
            run_experiment(args.format, optimizer, config_overrides)

    else:
        # Default: run BF16 baseline
        run_experiment("BF16", "adamw", config_overrides)


if __name__ == "__main__":
    main()
