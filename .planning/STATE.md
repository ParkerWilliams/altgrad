# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-20)

**Core value:** Evidence-backed answer to which 8-bit floating-point format most benefits from geometry-aware updates, and why.
**Current focus:** Phase 2 - Baseline Validation (COMPLETE - ready for RunPod deployment)

## Current Position

Phase: 2 of 6 (Baseline Validation) - COMPLETE
Plan: 5 of 5 in current phase
Status: Phase complete (pending RunPod execution)
Last activity: 2026-01-21 - Completed 02-05-PLAN.md (E5M2 FP8 Experiment)

Progress: [########..] 100% Phase 2

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 14 min
- Total execution time: 1.9 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-quantization-engine | 3 | 71 min | 24 min |
| 02-baseline-validation | 5 | 38 min | 8 min |

**Recent Trend:**
- Last 3 plans: 6 min, 8 min, 10 min
- Trend: Stable, fast execution

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
- **NEW:** Use lex_glue/ecthr_a instead of nlpaueb/multi_eurlex (dataset script format deprecated)
- **NEW:** Explicit step numbers in W&B logging (avoids drift)
- **NEW:** save_checkpoint on first NaN, stop at nan_patience
- **NEW:** CheckpointManager keeps best separately from rotation
- **NEW:** Flash Attention via scaled_dot_product_attention on PyTorch 2.0+
- **NEW:** SNR = |mean| / std for gradient signal quality measurement
- **NEW:** Experiment runner pattern: `python experiments/run_experiment.py <config>`
- **NEW:** 10M param model size (6 layers, 384 dim, 6 heads) for budget-constrained runs
- **NEW:** Simulated FP8 quantization (quantize/dequantize simulation on CPU/GPU without FP8 tensor cores)
- **NEW:** Local verification skipped for training runs - targeting H100 RunPod deployment

### Pending Todos

**RunPod Execution:**
1. Upload code to H100 RunPod instance
2. Run BF16 baseline: `python experiments/run_experiment.py experiments/configs/bf16_baseline.yaml`
3. Run E5M2 FP8: `python experiments/run_experiment.py experiments/configs/e5m2_fp8.yaml`
4. Generate comparison summary from W&B results

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

Last session: 2026-01-21
Stopped at: Completed 02-05-PLAN.md (E5M2 FP8 Experiment) - Phase 2 complete
Resume file: None

## Next Steps

1. **Deploy to RunPod:** Upload codebase to H100 instance
2. **Execute Experiments:** Run BF16 baseline and E5M2 FP8 experiments
3. **Phase 3 Planning:** Begin Model Integration phase (QuantizedLinear wrappers)
