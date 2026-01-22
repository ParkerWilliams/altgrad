---
phase: "04"
plan: "03"
subsystem: "training-infrastructure"
tags: ["fp8", "format-runner", "experiment-config", "stability"]
requires: ["04-01", "04-02"]
provides: ["format-configs", "experiment-runner", "failure-reports"]
affects: ["05-experiment-execution"]
tech-stack:
  added: []
  patterns: ["self-contained-runner", "failure-report-generation"]
key-files:
  created:
    - experiments/configs/e5m2_short.yaml
    - experiments/configs/e0m7_uniform.yaml
    - experiments/configs/e1m6_uniform.yaml
    - experiments/configs/e3m4_uniform.yaml
    - experiments/configs/e7m0_uniform.yaml
    - altgrad/training/format_runner.py
  modified:
    - altgrad/training/config.py
    - altgrad/training/__init__.py
decisions:
  - "500 steps for all format comparison configs (valid comparison)"
  - "Self-contained runner creates own Trainer (not extends)"
metrics:
  duration: "4 min"
  completed: "2026-01-22"
---

# Phase 04 Plan 03: Format Experiment Runner Summary

**One-liner:** Self-contained FormatExperimentRunner with stability interventions, diagnostic logging, and markdown failure report generation for exotic FP8 format testing.

## What Was Built

### Format Experiment Configs (5 files)
Created consistent experiment configurations for format comparison:

| Config | Format | Steps | Purpose |
|--------|--------|-------|---------|
| e5m2_short.yaml | E5M2 | 500 | Baseline for format comparison |
| e0m7_uniform.yaml | E0M7 | 500 | Fixed-point format test |
| e1m6_uniform.yaml | E1M6 | 500 | Two-scale format test |
| e3m4_uniform.yaml | E3M4 | 500 | Balanced format test |
| e7m0_uniform.yaml | E7M0 | 500 | Powers-of-2 format test (high-risk) |

All configs share: seed=42, batch_size=12, learning_rate=6e-4, use_shadow=true.

### TrainConfig Extensions
Added Phase 4 stability and diagnostic settings:

```python
# Stability interventions
enable_partition_clipping: bool = False  # STAB-05
partition_clip_base: float = 1.0
enable_emergency_shift: bool = False  # STAB-06
emergency_shift_nan_patience: int = 3
emergency_shift_stall_threshold: float = 0.5

# Diagnostic sampling
diagnostic_interval: int = 50
log_stiffness: bool = False
log_grid_alignment: bool = False
log_ulp: bool = False
```

### FormatExperimentRunner (761 lines)
Self-contained experiment runner that:

1. **Creates its own Trainer** - Not a subclass, fully self-contained
2. **Integrates stability interventions:**
   - PartitionRelativeClipper (STAB-05) for format-aware gradient clipping
   - EmergencyMantissaShift (STAB-06) for automatic format fallback
3. **Collects advanced diagnostics:**
   - Stiffness field statistics (DIAG-01)
   - Grid alignment metrics (DIAG-02)
   - ULP movement tracking (DIAG-04)
   - Gradient-stiffness correlation (DIAG-03)
4. **Generates failure reports** on collapse:
   - Exact collapse step and reason
   - Gradient sparsity analysis
   - Zero-update region detection
   - Diagnostic trend over last 5 measurements
   - Configuration dump
   - Recommendations based on failure mode

### Exported APIs

```python
from altgrad.training import (
    FormatExperimentRunner,  # Main runner class
    ExperimentResult,        # Result dataclass
    DiagnosticSnapshot,      # Diagnostic measurement
    run_format_experiment,   # Convenience function
)
```

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 9172f82 | feat | Add e5m2 short baseline and format-specific experiment configs |
| acdc93d | feat | Extend TrainConfig with stability and diagnostic settings |
| 3a16f93 | feat | Implement self-contained FormatExperimentRunner with failure report generation |

## Verification

```bash
# 7 config files exist (bf16_baseline, e5m2_fp8, e5m2_short, 4 format tests)
ls experiments/configs/*.yaml  # OK

# All format test configs use 500 steps
grep "max_steps: 500" experiments/configs/e*_uniform.yaml experiments/configs/e5m2_short.yaml  # All 5 match

# TrainConfig has new fields
python -c "from altgrad.training import TrainConfig; c = TrainConfig(enable_partition_clipping=True); print(c.enable_partition_clipping)"  # True

# FormatExperimentRunner imports correctly
python -c "from altgrad.training import FormatExperimentRunner, ExperimentResult, run_format_experiment; print('OK')"  # OK
```

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

Phase 4 complete. Ready for Phase 5 (Experiment Execution):
- All format configs created with consistent settings
- FormatExperimentRunner ready for H100 deployment
- Failure report generation will capture E7M0 collapse details
- Stability interventions available for recovery attempts

### Deployment Checklist
1. Upload codebase to H100 RunPod
2. Run experiments in order: e5m2_short -> e3m4 -> e1m6 -> e0m7 -> e7m0
3. Collect W&B logs and failure reports
4. Generate format comparison analysis
