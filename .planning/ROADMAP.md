# Roadmap: AltGrad

## Overview

AltGrad progresses from isolated quantization mechanics through validated baselines to exotic format testing and manifold-aware optimization. The journey prioritizes diagnostic depth over feature breadth: each phase delivers observable, testable capabilities that build toward answering "Which FP8 format most benefits from geometry-aware updates?" Phases 1-3 establish infrastructure, Phase 4 tests custom formats (with E7M0 as expected-failure experiment), Phase 5 integrates the novel optimizer, and Phase 6 synthesizes findings.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Quantization Engine** - Standalone FP8 format implementations with STE gradient flow
- [x] **Phase 2: Baseline Validation** - BF16 baseline and standard FP8 training with monitoring infrastructure
- [x] **Phase 3: Model Integration** - QuantizedLinear wrappers and nanoGPT surgery without forking
- [ ] **Phase 4: Custom Format Testing** - E1M6, E3M4, E7M0 experiments (E7M0 as negative result)
- [ ] **Phase 5: Manifold-Aware Optimizer** - Stiffness-preconditioned updates and geometry diagnostics
- [ ] **Phase 6: Analysis & Documentation** - Comparative analysis and failure mode documentation

## Phase Details

### Phase 1: Quantization Engine
**Goal**: Standalone quantization module that correctly implements all FP8 formats with gradient flow
**Depends on**: Nothing (first phase)
**Requirements**: QUANT-01, QUANT-02, QUANT-03, QUANT-04, STAB-04
**Success Criteria** (what must be TRUE):
  1. Each format (E0M7, E1M6, E3M4, E5M2, E7M0) correctly round-trips values within representable range
  2. STE gradient override passes gradients through quantize/dequantize unchanged
  3. Per-tensor scaling with amax history correctly tracks dynamic range across batches
  4. Transfer functions correctly map bit-indices to real values and back for each format
  5. Bit-stall detection correctly identifies when quantized updates round to zero
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — FP8 format registry with transfer functions (TDD)
- [x] 01-02-PLAN.md — STE quantize/dequantize operations (TDD)
- [x] 01-03-PLAN.md — Dynamic scaling and bit-stall detection

### Phase 2: Baseline Validation
**Goal**: Verified BF16 baseline and standard FP8 training with comprehensive monitoring infrastructure
**Depends on**: Phase 1
**Requirements**: METR-01, METR-02, METR-03, METR-04, METR-05, GRAD-01, GRAD-02, GRAD-03, GRAD-04, STAB-01, STAB-02, STAB-03, INTG-04, EXPR-01, EXPR-02, EXPR-03
**Success Criteria** (what must be TRUE):
  1. nanoGPT trains on EurLex in BF16 with stable loss curves (no divergence)
  2. All stability metrics (overflow/underflow/NaN counters, dynamic range) log to W&B per step
  3. Gradient statistics (norms, SNR, dead neurons, zero-update fraction) track per layer
  4. BF16 baseline comparison plots generate automatically after each run
  5. Checkpoint saves enable restart from any point within budget constraints
**Plans**: 5 plans in 4 waves

Plans:
- [x] 02-01-PLAN.md — EurLex data preparation (Wave 1)
- [x] 02-02-PLAN.md — Training infrastructure: config, metrics, checkpoints, W&B (Wave 1)
- [x] 02-03-PLAN.md — nanoGPT model and trainer with FP32 shadow (Wave 2)
- [x] 02-04-PLAN.md — BF16 baseline experiment (Wave 3)
- [x] 02-05-PLAN.md — E5M2 FP8 experiment with gradient comparison (Wave 4)

### Phase 3: Model Integration
**Goal**: QuantizedLinear wrappers inject quantization into nanoGPT without forking, supporting per-layer mixed precision
**Depends on**: Phase 2
**Requirements**: INTG-01, INTG-02, INTG-03, EXPR-04
**Success Criteria** (what must be TRUE):
  1. QuantizedLinear wrapper correctly quantizes forward/backward passes at layer boundaries
  2. quantize_model() replaces Linear layers post-init without modifying nanoGPT source
  3. Per-layer mixed precision config allows attention in BF16 while MLP uses FP8
  4. Format ablation runs produce identical seeds/ordering, varying only FP8 format
**Plans**: 2 plans in 2 waves

Plans:
- [x] 03-01-PLAN.md — QuantizedLinear wrapper and model surgery (INTG-01, INTG-02)
- [x] 03-02-PLAN.md — Per-layer mixed precision config and ablation reproducibility (INTG-03, EXPR-04)

### Phase 4: Custom Format Testing
**Goal**: Systematic testing of exotic formats, documenting E7M0 failure modes as scientific negative result
**Depends on**: Phase 3
**Requirements**: STAB-05, STAB-06, DIAG-01, DIAG-02, DIAG-03, DIAG-04
**Success Criteria** (what must be TRUE):
  1. E3M4 achieves convergence (loss decreases) comparable to E5M2 baseline
  2. E1M6 works on normalized tensors (post-LayerNorm) without overflow
  3. E7M0 failure mode documented: exact step where training collapses, gradient sparsity, zero-update regions
  4. Partition-relative gradient clipping adapts to each format's dynamic range
  5. Emergency mantissa shift triggers correctly on persistent NaN/high stall rates
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Manifold-Aware Optimizer
**Goal**: Stiffness-preconditioned optimizer that treats FP8 as geometric manifold, validated on viable formats
**Depends on**: Phase 4
**Requirements**: MANI-01, MANI-02, MANI-03, MANI-04
**Success Criteria** (what must be TRUE):
  1. Stiffness factor S = 2^(floor(log2|w|) - M) computes correctly for all weight tensors
  2. Stiffness-preconditioned updates move weights by consistent ULP counts (not real-value deltas)
  3. Standard vs manifold-aware toggle produces measurably different training dynamics
  4. Bit-position tracking shows latent integer state evolution across training steps
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Analysis & Documentation
**Goal**: Synthesis of findings answering which FP8 format benefits most from geometry-aware updates
**Depends on**: Phase 5
**Requirements**: ANAL-01, ANAL-02, ANAL-03
**Success Criteria** (what must be TRUE):
  1. Summary analysis identifies sweet-spot format per layer type (attention, MLP, classifier)
  2. Failure mode documentation shows where each format fails (forward, backward, optimizer)
  3. Manifold-aware vs standard comparison quantifies benefit (or lack thereof) for each format
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Quantization Engine | 3/3 | Complete | 2026-01-21 |
| 2. Baseline Validation | 5/5 | Complete | 2026-01-21 |
| 3. Model Integration | 2/2 | Complete | 2026-01-21 |
| 4. Custom Format Testing | 0/? | Not started | - |
| 5. Manifold-Aware Optimizer | 0/? | Not started | - |
| 6. Analysis & Documentation | 0/? | Not started | - |

---
*Roadmap created: 2026-01-20*
*Last updated: 2026-01-21*
