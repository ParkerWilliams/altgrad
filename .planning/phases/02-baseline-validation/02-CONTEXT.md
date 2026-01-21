# Phase 2: Baseline Validation - Context

**Gathered:** 2026-01-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Verified BF16 baseline and standard FP8 (E5M2) training on nanoGPT + EurLex with comprehensive monitoring infrastructure. Establishes the reference point against which all exotic formats (Phase 4) will be compared. Includes W&B logging, stability metrics, gradient statistics, and checkpoint management.

</domain>

<decisions>
## Implementation Decisions

### Metrics Granularity
- Log to W&B every step (full visibility for research)
- Gradient statistics tracked per-layer (norms, SNR, dead neurons)
- FP8 vs FP32 gradient cosine similarity via continuous shadow (FP32 copy maintained for comparison every step)
- Track ALL stability metrics: overflow/underflow counts, bit-stall rates, dynamic range utilization

### Baseline Experiment Design
- Identical random seeds for BF16 and FP8 runs (differences purely from precision)
- Auto-generate comparison plots after each run via W&B dashboard
- Run until convergence trend visible (not fixed step count)
- Validate both BF16 baseline AND E5M2 FP8 in this phase (confirm standard FP8 works before exotic formats)

### Checkpoint Strategy
- Save every 100 steps
- Retain: best validation checkpoint + last 2
- Include: model weights + optimizer state + quantization state (amax history, scale factors)
- Storage: local disk only
- W&B resume: continue same run on restart (seamless history)
- Auto-save checkpoint on stability anomaly (NaN detected, etc.) for debugging

### Stability Thresholds
- NaN handling: log and continue, auto-stop after 10 consecutive NaN steps
- Divergence detection: patience window (flag if loss increases N consecutive steps, don't auto-stop)
- Bit-stall warning: trigger at >50% stall rate
- Overflow/underflow warning: trigger at >1% of values clipped
- Dead neuron tracking: per-layer, defined as gradient below threshold (1e-8) for N steps
- Zero-update fraction: tracked as distinct key metric (weights unchanged despite gradient)
- Dynamic range utilization: histogram of used bit patterns
- Warning action: log as W&B alert (never auto-stop for warnings, only for persistent NaN)

### Claude's Discretion
- Exact W&B dashboard layout
- Specific patience window sizes for divergence detection
- EurLex data loading and batching details
- nanoGPT integration patterns

</decisions>

<specifics>
## Specific Ideas

- Continuous FP32 shadow for gradient comparison — full fidelity, not sampling
- W&B alerts for threshold breaches (not just logging)
- Bit pattern histogram to spot wasted FP8 capacity
- Save-on-anomaly for post-mortem debugging of failures

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-baseline-validation*
*Context gathered: 2026-01-21*
