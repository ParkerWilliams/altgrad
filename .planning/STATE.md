# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-20)

**Core value:** Evidence-backed answer to which 8-bit floating-point format most benefits from geometry-aware updates, and why.
**Current focus:** Phase 3 - Model Integration (in progress)

## Current Position

Phase: 3 of 6 (Model Integration)
Plan: 1 of 3 in current phase
Status: In progress
Last activity: 2026-01-21 - Completed 03-01-PLAN.md (QuantizedLinear and Surgery)

Progress: [#########.] 90% Phase 2-3 (9/10 plans through Phase 3)

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 13 min
- Total execution time: 2.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-quantization-engine | 3 | 71 min | 24 min |
| 02-baseline-validation | 5 | 38 min | 8 min |
| 03-model-integration | 1 | 3 min | 3 min |

**Recent Trend:**
- Last 3 plans: 8 min, 10 min, 3 min
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
- **NEW:** Skip patterns approach for weight tying preservation (skip lm_head)
- **NEW:** In-place surgery modifies model directly (no copy)
- **NEW:** Collect-then-modify pattern to avoid mutating during iteration
- **NEW:** QuantizedLinear exposes weight/bias properties for optimizer compatibility

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
Stopped at: Completed 03-01-PLAN.md (QuantizedLinear and Surgery)
Resume file: None

## Next Steps

1. **Continue Phase 3:** Execute 03-02-PLAN.md (FP32 Shadow Model)
2. **Deploy to RunPod:** Upload codebase to H100 instance for experiments
3. **Execute Experiments:** Run BF16 baseline and E5M2 FP8 experiments
