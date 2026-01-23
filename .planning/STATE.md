# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-20)

**Core value:** Evidence-backed answer to which 8-bit floating-point format most benefits from geometry-aware updates, and why.
**Current focus:** Phase 5 - Manifold-Aware Optimizer (IN PROGRESS)

## Current Position

Phase: 5 of 6 (Manifold-Aware Optimizer)
Plan: 2 of 3 in current phase
Status: In progress
Last activity: 2026-01-22 - Completed 05-02-PLAN.md (TrainConfig Integration)

Progress: [###############] 100% (15/16 plans through Phase 5.2)

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Average duration: 10 min
- Total execution time: 2.6 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-quantization-engine | 3 | 71 min | 24 min |
| 02-baseline-validation | 5 | 38 min | 8 min |
| 03-model-integration | 2 | 9 min | 5 min |
| 04-custom-format-testing | 3 | 12 min | 4 min |
| 05-manifold-aware-optimizer | 2 | 11 min | 6 min |

**Recent Trend:**
- Last 3 plans: 4 min, 8 min, 3 min
- Trend: Fast execution continuing

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- E3M4 bias=1 for range ~0.06 to ~124 (plan had bias=7 which gave wrong range)
- Round-to-nearest-even for FP8 to_bits() tie-breaking (IEEE 754 standard)
- E0M7 as pure fixed-point with sign-magnitude representation in [-127/128, 127/128]
- STE passes gradients unchanged through quantization (dx = dy) - enables gradient flow through non-differentiable quantization
- Dequantize scales gradient by scale factor (chain rule)
- Scale parameter non-differentiable (returns None in backward pass)
- Use lex_glue/ecthr_a instead of nlpaueb/multi_eurlex (dataset script format deprecated)
- Explicit step numbers in W&B logging (avoids drift)
- save_checkpoint on first NaN, stop at nan_patience
- CheckpointManager keeps best separately from rotation
- Flash Attention via scaled_dot_product_attention on PyTorch 2.0+
- SNR = |mean| / std for gradient signal quality measurement
- Experiment runner pattern: `python experiments/run_experiment.py <config>`
- 10M param model size (6 layers, 384 dim, 6 heads) for budget-constrained runs
- Simulated FP8 quantization (quantize/dequantize simulation on CPU/GPU without FP8 tensor cores)
- Local verification skipped for training runs - targeting H100 RunPod deployment
- Skip patterns approach for weight tying preservation (skip lm_head)
- In-place surgery modifies model directly (no copy)
- Collect-then-modify pattern to avoid mutating during iteration
- QuantizedLinear exposes weight/bias properties for optimizer compatibility
- First-match-wins for LayerPrecisionRule pattern matching
- format=None in config means keep layer in BF16 (not quantize)
- quantize_model requires exactly one of format or config (mutex)
- E5M2_MAX=57344 as baseline for partition-relative threshold scaling
- 1% overflow threshold for clipper activation
- 3 consecutive NaNs or >50% stall rate triggers emergency format shift
- Fallback chain: E7M0->E5M2, E1M6->E3M4, E0M7->E3M4, E3M4->E5M2, E5M2->None
- E0M7 constant 1/128 stiffness (uniform grid spacing)
- torch.nextafter for IEEE 754 compliant ULP computation
- NaN stiffness for zero weights (undefined)
- 500 steps for all format comparison configs (valid comparison)
- Self-contained runner creates own Trainer (not extends)
- **NEW:** Stiffness preconditioning BEFORE momentum (grad * stiffness, then exp_avg update)
- **NEW:** NaN stiffness replaced with 1.0 neutral multiplier
- **NEW:** Bit-position tracks signed ULP movement (direction-aware)
- **NEW:** use_manifold_aware toggle in TrainConfig controls optimizer selection
- **NEW:** log_bit_position=True logs mean/std/min/max bit-position stats
- **NEW:** manifold_mantissa_bits default=2 matches E5M2 format

### Pending Todos

**RunPod Execution:**
1. Upload code to H100 RunPod instance
2. Run BF16 baseline: `python experiments/run_experiment.py experiments/configs/bf16_baseline.yaml`
3. Run E5M2 short: `python experiments/run_experiment.py experiments/configs/e5m2_short.yaml`
4. Run format experiments: e3m4_uniform, e1m6_uniform, e0m7_uniform, e7m0_uniform
5. Collect W&B logs and failure reports
6. Generate format comparison analysis

### Blockers/Concerns

**High-risk experiments:**
- E7M0 has >90% probability of training failure (expected negative result)
- E0M7 narrow dynamic range may cause overflow on unnormalized tensors

**Budget constraint:**
- Single H100, $20 total compute limits experiment iterations
- Short runs only - convergence trends, not full training

### Quick Tasks Completed

| # | Description | Date | Directory |
|---|-------------|------|-----------|
| 001 | Add virtual environment note to PROJECT.md | 2026-01-20 | [001-virtual-environment-note](./quick/001-virtual-environment-note/) |

## Session Continuity

Last session: 2026-01-22
Stopped at: Completed 05-02-PLAN.md (TrainConfig Integration)
Resume file: None

## Next Steps

1. **Continue Phase 5:** Plan 05-03 (Manifold experiment validation)
2. **Deploy to RunPod:** Upload codebase to H100 instance for experiments
3. **Run manifold experiments:** Compare e5m2_manifold vs e5m2_standard
4. **Run format experiments:** Compare all FP8 formats with manifold-aware training
