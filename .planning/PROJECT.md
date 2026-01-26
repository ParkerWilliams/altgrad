# AltGrad

## What This Is

A small-scale experimental test bench for studying how different 8-bit floating-point datatypes (varying mantissa/exponent splits) affect transformer training stability, accuracy, and gradient behavior. The core innovation is a **manifold-aware optimizer** that treats the FP8 representation as a warped manifold of uniform bit-space, enabling updates that move by fixed ULPs rather than fixed real values.

Built on nanoGPT, tested on EurLex legal documents, constrained to single H100 and $20 compute.

## Core Value

**Evidence-backed answer to:** "Which 8-bit floating-point format most benefits from geometry-aware updates, and why?"

If everything else fails, the test bench must produce clear data showing where each format fails (forward precision, gradient fidelity, or optimizer update) and whether the manifold-aware step delays or prevents that failure.

## Current State

**v1.0 shipped:** 2026-01-26

Infrastructure complete and ready for H100 experiments:
- 5 FP8 formats (E0M7, E1M6, E3M4, E5M2, E7M0) with STE gradient flow
- nanoGPT trainer with EurLex dataset, W&B logging, checkpointing
- QuantizedLinear wrappers with per-layer mixed precision
- ManifoldAdamW and GridOptim optimizers
- Flip metrics, rank health monitoring, stall ratio tracking
- Analysis pipeline with report generation

**Stats:** 13,416 lines Python, 18 source files, 319 tests passing

## Requirements

### Validated

- FP8 format registry (E0M7, E1M6, E3M4, E5M2, E7M0) — v1.0
- Quantize/dequantize with STE gradient override — v1.0
- Per-tensor scaling with amax history — v1.0
- Stability monitoring (overflow/underflow/NaN counters) — v1.0
- Gradient statistics (norms, SNR, dead neurons) — v1.0
- QuantizedLinear wrapper with model surgery — v1.0
- Per-layer mixed precision config — v1.0
- EurLex dataset integration — v1.0
- ManifoldAdamW with stiffness preconditioning — v1.0
- GridOptim with stochastic rounding — v1.0
- Flip metrics and rank health monitoring — v1.0
- Experiment config grid with W&B logging — v1.0
- Analysis reports (format comparison, failure modes, manifold comparison) — v1.0

### Active

- [ ] Run BF16 baseline experiment on H100
- [ ] Run format comparison experiments (E5M2, E3M4, E1M6, E0M7, E7M0)
- [ ] Run manifold-aware vs standard A/B comparison
- [ ] Regenerate analysis reports with real experiment data
- [ ] Document findings answering core research question

### Out of Scope

- Large-scale pretraining — diagnostic insight only, not SOTA
- Multi-GPU scaling — single H100 constraint
- Inference-only optimization — training focus
- Gradient-only quantization — defer unless needed
- Optimizer state quantization — defer unless needed
- Full convergence runs — short runs sufficient for trend visibility

## Context

### The Manifold-Aware Update

Standard gradient descent: `w_new = w_old - lr * grad` applies uniform steps in real-number space.

The manifold-aware approach treats FP8 bit-strings as a uniform integer domain (0-255) warped onto reals through a format-specific transfer function:

1. **Stiffness Factor:** `S = 2^(floor(log2|w|) - M)` where M = mantissa bits
2. **Effective Gradient:** `grad_effective = grad_w * S`
3. **Update:** Moves weight by fixed number of ULPs, not fixed reals

This is essentially a Natural Gradient where the metric is the floating-point density itself. It should particularly help:
- Near zero (where ULPs are dense)
- In E7M0 (powers-of-2 only) where standard updates would move multiple magnitude steps unknowingly

### E7M0: The Logarithmic Limit

E7M0 (7 exponent bits, 0 mantissa bits) represents only powers of 2. This is the extreme stress test:
- Maximum stiffness (jumping from 1.0 to 2.0 is 1 ULP)
- No mantissa interpolation
- Tests whether manifold-aware updates can navigate a purely exponential weight space

If the optimizer converges here, it proves the transformation effectively flattens the exponential manifold into a searchable uniform space.

### Dataset: EurLex

Legal document classification with hierarchical categories. Chosen over standard benchmarks (WikiText, TinyStories) because:
- "Jagged" loss landscapes from fine semantic distinctions
- Long-context gradients for stability testing
- Directly applicable to production document classifiers

### Technical Approach

- **Backbone:** nanoGPT (vanilla PyTorch, exposed training loop)
- **Quantization:** Simulated (store FP32/BF16, quantize at boundaries)
- **Injection point:** Stiffness calculation in optimizer.step()
- **State tracking:** Bit-position as latent integer state

## Development Environment

All Python operations must run within a virtual environment:

```bash
# Create (once)
python -m venv .venv

# Activate (each session)
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Constraints

- **Hardware:** Single NVIDIA H100
- **Budget:** $20 total compute
- **Model size:** 50M parameters
- **Runtime:** Short runs (convergence trend, not full training)
- **Primary goal:** Diagnostic insight, not absolute performance

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| nanoGPT over HuggingFace | Exposed training loop, vanilla PyTorch, easy optimizer injection | Good — easy integration |
| EurLex over WikiText/TinyStories | Jagged loss landscapes, real-world applicability, long context | Good — 18.7M train tokens |
| Simulated FP8 over native kernels | Faster dev iteration, budget constraint, sufficient for diagnostic | Good — works on any GPU |
| E7M0 in v1 scope | Logarithmic limit validates stiffness theory at extreme | Pending experiment |
| Mixed precision by layer in v1 | Key hypothesis: attention vs classifier head have different precision needs | Pending experiment |
| Stiffness formula: 2^(floor(log2\|w\|) - M) | Captures mantissa-dependent ULP spacing, accounts for exponent | Implemented in ManifoldAdamW |
| E3M4 bias=1 | Range ~0.06 to ~124, covers typical weight magnitudes | Good |
| Round-to-nearest-even | IEEE 754 standard for tie-breaking | Good |
| Stiffness preconditioning before momentum | Ensures ULP-consistent updates | Implemented |
| Stall ratio = 1 - (flips/updates) | Clear metric for gradient effectiveness | Good |

---
*Last updated: 2026-01-26 after v1.0 milestone*
