---
phase: 02-baseline-validation
plan: 02
subsystem: training-infrastructure
tags: [config, metrics, checkpoint, wandb, testing]

dependency-graph:
  requires: [01-quantization-engine]
  provides: [training-config, gradient-metrics, checkpoint-management, wandb-tracking]
  affects: [02-03, 02-04, 03-geometry-baselines]

tech-stack:
  added: [pyyaml, wandb]
  patterns: [dataclass-config, explicit-step-logging, checkpoint-rotation]

key-files:
  created:
    - altgrad/training/config.py
    - altgrad/training/metrics.py
    - altgrad/training/checkpoint.py
    - altgrad/training/callbacks.py
    - tests/test_training_infra.py
  modified:
    - altgrad/training/__init__.py

decisions:
  - id: dec-0202-1
    choice: "Explicit step numbers in W&B logging"
    rationale: "Avoids drift between W&B step and actual training step"
  - id: dec-0202-2
    choice: "save_checkpoint on first NaN, stop at nan_patience"
    rationale: "Capture state before divergence for debugging"
  - id: dec-0202-3
    choice: "CheckpointManager keeps best separately from rotation"
    rationale: "Never lose best model even with small max_checkpoints"

metrics:
  duration: 6 min
  completed: 2026-01-21
---

# Phase 2 Plan 2: Training Infrastructure Summary

**One-liner:** TrainConfig dataclass with YAML I/O, per-layer gradient metrics, checkpoint rotation with best tracking, and W&B alerts for NaN/bit-stall/overflow thresholds.

## What Was Built

### 1. Configuration Module (`config.py`)
- `TrainConfig` dataclass with 30+ fields covering model, training, quantization, checkpointing, stability thresholds, logging, W&B, and seed
- `save_config(config, path)` - YAML serialization with directory creation
- `load_config(path)` - YAML deserialization to TrainConfig

### 2. Metrics Module (`metrics.py`)
- `compute_gradient_stats(model, threshold)` - Per-layer L2/Linf norms, dead neuron fraction, SNR with aggregates
- `compute_stability_metrics(model, detector)` - NaN/Inf counts, optional bit_stall_rate
- `gradient_cosine_similarity(model_a, model_b)` - Per-layer cosine similarity for comparing FP32 vs FP8

### 3. Checkpoint Module (`checkpoint.py`)
- `save_checkpoint(filepath, model, optimizer, scaler, step, config, quantization_state)` - Full state including RNG
- `load_checkpoint(filepath, model, optimizer, scaler)` - Restore everything including RNG
- `CheckpointManager` - Automatic rotation (keep N most recent + best), anomaly saves

### 4. Callbacks Module (`callbacks.py`)
- `WandbTracker(config, run_id)` - W&B initialization with resume support
- `log_step(step, metrics)` - Explicit step numbers in all logs
- `check_alerts(step, metrics, config)` - Returns 'continue', 'save_checkpoint', or 'stop'
- Alerts: NaN patience, bit_stall_threshold, overflow_threshold

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 464ca00 | feat | TrainConfig and gradient metrics modules |
| 44c1faa | feat | Checkpoint manager and W&B tracker |
| 4b8449b | test | Training infrastructure tests (20 tests) |
| 37b43e7 | fix | Restore __init__.py exports after merge conflict |

## Test Coverage

20 tests across 4 test classes:
- `TestConfig` (3): defaults, YAML roundtrip, FP8 options
- `TestGradientStats` (6): shapes, L2 norm, dead fraction, SNR, cosine similarity identical/opposite
- `TestStabilityMetrics` (2): no issues, with detector
- `TestCheckpoint` (3): save/load, RNG state, quantization state
- `TestCheckpointManager` (2): max checkpoints rotation, best tracking
- `TestWandbTracker` (4): init, log_step, NaN alerts, bit stall alerts

All tests use mocking for W&B to avoid actual API calls.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Merge conflict with concurrent plan execution**
- **Found during:** Verification step
- **Issue:** Plan 02-01 was executing concurrently and overwrote `__init__.py` with older version lacking new exports
- **Fix:** Re-applied the exports for config, metrics, checkpoint, callbacks
- **Files modified:** altgrad/training/__init__.py
- **Commit:** 37b43e7

## Verification Results

```
1. from altgrad.training import TrainConfig, load_config, save_config - OK
2. from altgrad.training import compute_gradient_stats, compute_stability_metrics - OK
3. from altgrad.training import save_checkpoint, load_checkpoint, CheckpointManager - OK
4. from altgrad.training import WandbTracker - OK
5. pytest tests/test_training_infra.py - 20 passed
```

## Success Criteria Met

- [x] Configuration loads from YAML and supports all required fields
- [x] Gradient statistics compute per-layer metrics correctly
- [x] Checkpoints save/restore full training state including RNG
- [x] W&B tracker logs with explicit steps and fires alerts on thresholds
- [x] All infrastructure tested without requiring actual W&B connection

## Next Phase Readiness

Plan 02-02 provides the infrastructure that 02-03 (GPT model) and 02-04 (training loop) depend on:
- `TrainConfig` will configure model architecture and training hyperparameters
- `compute_gradient_stats` will feed metrics to `WandbTracker.log_step`
- `CheckpointManager` will be used in training loop for periodic saves
- `check_alerts` will trigger early stopping or anomaly saves
