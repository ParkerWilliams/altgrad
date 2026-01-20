# Requirements: AltGrad

**Defined:** 2026-01-20
**Core Value:** Evidence-backed answer to which 8-bit floating-point format most benefits from geometry-aware updates, and why.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Quantization Engine

- [ ] **QUANT-01**: FP8 format registry supporting E0M7, E1M6, E3M4, E5M2, E7M0 specifications
- [ ] **QUANT-02**: Quantize/dequantize functions with Straight-Through Estimator gradient override
- [ ] **QUANT-03**: Per-tensor scaling with delayed amax tracking (history buffer)
- [ ] **QUANT-04**: Format-specific transfer functions (bit-index <-> real value)

### Stability Monitoring

- [ ] **STAB-01**: Per-tensor overflow/underflow counters (forward, backward, optimizer)
- [ ] **STAB-02**: NaN/Inf detection with configurable early stopping
- [ ] **STAB-03**: Dynamic range tracking (amax moving average)
- [ ] **STAB-04**: Bit-Stall counter (tracks when `round(b + delta_b) == b` despite non-zero gradient)
- [ ] **STAB-05**: Partition-relative gradient clipping (clip based on format's dynamic range)
- [ ] **STAB-06**: Emergency mantissa shift (increase M bits on persistent NaN/high stall rate)

### Training Metrics

- [ ] **METR-01**: Train/validation loss logging per step/epoch
- [ ] **METR-02**: Perplexity tracking for language modeling quality
- [ ] **METR-03**: Wall-clock time and throughput (tokens/sec)
- [ ] **METR-04**: BF16 baseline comparison plots (automated)
- [ ] **METR-05**: Gradient cosine similarity (FP8 vs FP32 reference, periodic)

### Gradient Statistics

- [ ] **GRAD-01**: Per-layer gradient norms (L2, Linf)
- [ ] **GRAD-02**: Gradient SNR (FP8 vs FP32 reference)
- [ ] **GRAD-03**: Dead neuron fraction detection
- [ ] **GRAD-04**: Zero-update fraction tracking (weights that didn't change)

### Model Integration

- [ ] **INTG-01**: QuantizedLinear wrapper layer (inject quantization without forking nanoGPT)
- [ ] **INTG-02**: `quantize_model()` surgery function for post-init layer replacement
- [ ] **INTG-03**: Per-layer mixed precision config (attention in BF16, MLP in FP8)
- [ ] **INTG-04**: EurLex dataset integration with nanoGPT training loop

### Manifold-Aware Optimizer

- [ ] **MANI-01**: Stiffness factor calculation: `S = 2^(floor(log2|w|) - M)`
- [ ] **MANI-02**: Stiffness-preconditioned gradient step (`grad *= S` before update)
- [ ] **MANI-03**: Standard vs manifold-aware training mode toggle
- [ ] **MANI-04**: Bit-position tracking (latent integer state for each weight)

### Manifold Diagnostics

- [ ] **DIAG-01**: Stiffness field visualization (per-weight S values over training)
- [ ] **DIAG-02**: Quantization grid alignment measurement (distance to nearest FP8 value)
- [ ] **DIAG-03**: Gradient-stiffness correlation analysis
- [ ] **DIAG-04**: ULP statistics (how many bit-positions each update moves)

### Experiment Infrastructure

- [ ] **EXPR-01**: YAML/JSON experiment config grid
- [ ] **EXPR-02**: Per-run logging (W&B integration)
- [ ] **EXPR-03**: Checkpoint management with restart capability
- [ ] **EXPR-04**: Format ablation runs (identical seeds/ordering across formats)

### Analysis Output

- [ ] **ANAL-01**: Summary analysis identifying sweet-spot format per layer type
- [ ] **ANAL-02**: Failure mode documentation (where each format fails: forward, backward, optimizer)
- [ ] **ANAL-03**: Manifold-aware vs standard comparison report

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Diagnostics

- **DIAG-05**: Hessian eigenvalue spectrum (expensive, sample periodically)
- **DIAG-06**: Bit-space uniformity analysis (delta_bit vs delta_w tracking)
- **DIAG-07**: Format capacity utilization (which FP8 values actually used)

### Automation

- **AUTO-01**: Format-specific stability threshold detection (auto LR sweep)
- **AUTO-02**: Comparative dynamics dashboard (live parallel coordinates)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Distributed training | Single H100 constraint |
| Production/inference optimization | Research test bench, not deployment |
| Automatic hyperparameter tuning | $20 budget limits sweep runs |
| Multiple optimizer baselines | Focus on AdamW + manifold-aware only |
| Data preprocessing pipelines | Use EurLex as-is to avoid confounds |
| Architectural stability hacks (SmoothSwiGLU, etc.) | Want raw format behavior, not masked |
| Full convergence runs | Short runs sufficient for trend visibility |
| Gradient-only quantization | Defer unless needed |
| Optimizer state quantization | Defer unless needed |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| QUANT-01 | Phase 1 | Pending |
| QUANT-02 | Phase 1 | Pending |
| QUANT-03 | Phase 1 | Pending |
| QUANT-04 | Phase 1 | Pending |
| STAB-01 | Phase 2 | Pending |
| STAB-02 | Phase 2 | Pending |
| STAB-03 | Phase 2 | Pending |
| STAB-04 | Phase 1 | Pending |
| STAB-05 | Phase 4 | Pending |
| STAB-06 | Phase 4 | Pending |
| METR-01 | Phase 2 | Pending |
| METR-02 | Phase 2 | Pending |
| METR-03 | Phase 2 | Pending |
| METR-04 | Phase 2 | Pending |
| METR-05 | Phase 2 | Pending |
| GRAD-01 | Phase 2 | Pending |
| GRAD-02 | Phase 2 | Pending |
| GRAD-03 | Phase 2 | Pending |
| GRAD-04 | Phase 2 | Pending |
| INTG-01 | Phase 3 | Pending |
| INTG-02 | Phase 3 | Pending |
| INTG-03 | Phase 3 | Pending |
| INTG-04 | Phase 2 | Pending |
| MANI-01 | Phase 5 | Pending |
| MANI-02 | Phase 5 | Pending |
| MANI-03 | Phase 5 | Pending |
| MANI-04 | Phase 5 | Pending |
| DIAG-01 | Phase 4 | Pending |
| DIAG-02 | Phase 4 | Pending |
| DIAG-03 | Phase 4 | Pending |
| DIAG-04 | Phase 4 | Pending |
| EXPR-01 | Phase 2 | Pending |
| EXPR-02 | Phase 2 | Pending |
| EXPR-03 | Phase 2 | Pending |
| EXPR-04 | Phase 3 | Pending |
| ANAL-01 | Phase 6 | Pending |
| ANAL-02 | Phase 6 | Pending |
| ANAL-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-01-20*
*Last updated: 2026-01-20 after roadmap creation*
