# AltGrad

## What This Is

A small-scale experimental test bench for studying how different 8-bit floating-point datatypes (varying mantissa/exponent splits) affect transformer training stability, accuracy, and gradient behavior. The core innovation is a **manifold-aware optimizer** that treats the FP8 representation as a warped manifold of uniform bit-space, enabling updates that move by fixed ULPs rather than fixed real values.

Built on nanoGPT, tested on EurLex legal documents, constrained to single H100 and ≤$20 compute.

## Core Value

**Evidence-backed answer to:** "Which 8-bit floating-point format most benefits from geometry-aware updates, and why?"

If everything else fails, the test bench must produce clear data showing where each format fails (forward precision, gradient fidelity, or optimizer update) and whether the manifold-aware step delays or prevents that failure.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Custom FP8 implementations for E1M6, E3M4, E5M2, E7M0 formats
- [ ] Simulated quantization (store FP32/BF16, quantize at forward/backward boundaries)
- [ ] Stiffness factor calculation: `S = 2^(floor(log2|w|) - M)`
- [ ] Manifold-aware optimizer step injected at optimizer.step()
- [ ] Standard vs manifold-aware training mode toggle
- [ ] Mixed precision by layer (attention vs classifier head)
- [ ] EurLex dataset integration with nanoGPT
- [ ] Metrics collection: F1, gradient norms, zero-update fraction, ULP statistics
- [ ] Gradient cosine similarity vs FP32 reference (periodic comparison)
- [ ] Overflow/underflow counters (NaN, Inf, flush-to-zero)
- [ ] Experiment config grid (YAML/JSON)
- [ ] Per-run logging (CSV or W&B)
- [ ] Summary analysis identifying sweet-spot format and failure modes

### Out of Scope

- Large-scale pretraining — diagnostic insight only, not SOTA
- Multi-GPU scaling — single H100 constraint
- Inference-only optimization — training focus
- DCSBM synthetic data — real-world "jagged" loss landscapes required
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
- **Budget:** ≤$20 total compute
- **Model size:** ≤50M parameters
- **Runtime:** Short runs (convergence trend, not full training)
- **Primary goal:** Diagnostic insight, not absolute performance

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| nanoGPT over HuggingFace | Exposed training loop, vanilla PyTorch, easy optimizer injection | — Pending |
| EurLex over WikiText/TinyStories | Jagged loss landscapes, real-world applicability, long context | — Pending |
| Simulated FP8 over native kernels | Faster dev iteration, budget constraint, sufficient for diagnostic | — Pending |
| E7M0 in v1 scope | Logarithmic limit validates stiffness theory at extreme | — Pending |
| Mixed precision by layer in v1 | Key hypothesis: attention vs classifier head have different precision needs | — Pending |
| Stiffness formula: 2^(floor(log2\|w\|) - M) | Captures mantissa-dependent ULP spacing, accounts for exponent | — Pending |

---
*Last updated: 2025-01-20 after initialization*
