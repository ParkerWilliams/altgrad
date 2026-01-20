# Features Research: FP8 Quantized Training Test Bench

**Domain:** Low-precision transformer training with exotic FP8 formats
**Researched:** 2026-01-20
**Overall confidence:** MEDIUM-HIGH

## Executive Summary

FP8 training test benches require a careful balance between diagnostic depth and experimental clarity. Based on ecosystem research, successful low-precision training frameworks share common table stakes (overflow/underflow monitoring, loss tracking, checkpointing) but differentiate through specialized diagnostics tailored to their research hypotheses.

For a manifold-aware optimizer testing exotic FP8 formats (E1M6, E3M4, E5M2, E7M0), the critical differentiation lies in **geometry-aware diagnostics** that surface how quantization warps the optimization landscape, not just whether training converges.

**Key insight from 2025-2026 research:** Standard FP8 work focuses on E4M3/E5M2 stability. Testing extreme formats like E7M0 requires fundamentally different instrumentation—you need to measure the manifold curvature induced by quantization, not just numerical overflow rates.

## Table Stakes Features

Essential features for valid low-precision training experiments. Missing any of these undermines experimental credibility.

### 1. Numerical Stability Monitoring
**What:** Track overflow, underflow, and NaN occurrences across all tensors
**Why expected:** FP8's limited dynamic range (e.g., ±448 for E4M3, ±57,344 for E5M2) makes overflow/underflow the primary failure mode. Standard E4M3 training diverges within thousands of steps without mitigation.
**Complexity:** Medium
**Implementation:**
- Per-tensor overflow/underflow counters (forward, backward, optimizer state)
- Dynamic range tracking: `amax` (absolute maximum) history per tensor
- NaN/Inf detection with early stopping
- Histogram binning to show how many values cluster near representable limits

**Sources:**
- [NVIDIA FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) - E4M3/E5M2 range specifications
- [ICLR 2026 FP8 Stability](https://arxiv.org/pdf/2511.05811) - Standard FP8 diverges without outlier suppression
- [FP8 Mixed-Precision Framework](https://www.emergentmind.com/topics/fp8-mixed-precision-training-framework) - Auto-scaling for overflow/underflow mitigation

**Confidence:** HIGH (verified across multiple authoritative sources)

---

### 2. Training Loss & Metrics Tracking
**What:** Log train/val loss, perplexity, and convergence metrics every N steps
**Why expected:** Basic validation that training isn't silently degrading. Comparison baseline for format ablations.
**Complexity:** Low
**Implementation:**
- Per-step train loss
- Per-epoch validation loss and perplexity
- Learning rate schedule tracking
- Wall-clock time and throughput (tokens/sec)
- Comparison plots: BF16 baseline vs each FP8 format

**Sources:**
- [Neptune AI Model Training Report 2025](https://neptune.ai/state-of-foundation-model-training-report) - Standard metrics for foundation model training
- [W&B Sweeps Documentation](https://docs.wandb.ai/models/sweeps) - Hyperparameter sweep visualization patterns

**Confidence:** HIGH (industry-standard practice)

---

### 3. Gradient Statistics
**What:** Mean, variance, norm of gradients per layer
**Why expected:** Gradient explosion/vanishing is exacerbated in low-precision. Essential for debugging optimizer behavior.
**Complexity:** Medium
**Implementation:**
- Per-layer gradient norms (L2, Linf)
- Gradient histogram logging (detect dead neurons, saturation)
- Signal-to-noise ratio (SNR) of gradients in FP8 vs FP32 reference
- Gradient clipping events (when clipping triggers)

**Sources:**
- [Numerical Stability in Deep Learning](http://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html) - Gradient vanishing/explosion fundamentals
- [FP8-LM Training Paper](https://arxiv.org/pdf/2310.18313) - Gradient quantization strategies, SNR analysis

**Confidence:** HIGH (well-established diagnostic)

---

### 4. Checkpoint Management
**What:** Save model state periodically; enable restart from failure
**Why expected:** Single H100, ≤$20 budget means spot instance risk. Checkpoints prevent catastrophic loss of progress.
**Complexity:** Medium
**Implementation:**
- Periodic checkpointing (every N steps or on validation improvement)
- Lightweight state saving: model weights, optimizer state, RNG seeds
- Restart capability with exact reproducibility
- Checkpoint size monitoring (quantized states should be smaller)

**Sources:**
- [Just-In-Time Checkpointing (EuroSys 2024)](https://dl.acm.org/doi/10.1145/3627703.3650085) - Low-cost recovery for training failures
- [RepDL Library](https://www.webpronews.com/microsofts-repdl-open-source-library-boosts-deep-learning-reproducibility/) - Reproducibility via checkpointing

**Confidence:** HIGH (critical for budget-constrained research)

---

### 5. Format Ablation Support
**What:** Train identical model with different FP8 formats (E1M6, E3M4, E5M2, E7M0) and compare
**Why expected:** Core research question is format comparison. Must control all variables except mantissa/exponent split.
**Complexity:** Low-Medium
**Implementation:**
- Configuration file per format (mantissa bits, exponent bits)
- Shared codebase with format as parameter
- Parallel runs with identical seeds, data ordering, hyperparameters
- Automated comparison reports (convergence, final loss, stability)

**Sources:**
- [Ablation Studies in ML](https://www.baeldung.com/cs/ml-ablation-study) - Component removal methodology
- [AutoAblation Framework](https://dl.acm.org/doi/10.1145/3437984.3458834) - Parallel ablation execution

**Confidence:** HIGH (fundamental to comparative research)

---

### 6. Weight Distribution Tracking
**What:** Histograms of weight values over training
**Why expected:** Quantization's impact depends on weight distribution. Need to see if weights cluster in representable regions or hit quantization boundaries.
**Complexity:** Low
**Implementation:**
- Per-layer weight histograms (log scale to capture outliers)
- Percentage of weights at representable boundaries (e.g., max FP8 value)
- Divergence from BF16 weight distribution (KL divergence)
- Track weight staleness (how many unchanged across updates)

**Sources:**
- [Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Block-wise quantization analysis
- [Low-Bit Quantization Research (ACL 2025)](https://aclanthology.org/2025.acl-long.1555.pdf) - Quantization-induced degradation patterns

**Confidence:** MEDIUM-HIGH (standard for quantization research)

---

## Differentiators

Features specific to manifold-aware optimization and exotic FP8 format testing. These make this test bench novel.

### 7. Manifold Geometry Diagnostics
**What:** Measure curvature and distortion of the optimization landscape induced by quantization
**Why valuable:** Core hypothesis is that FP8 quantization creates a warped manifold. Standard metrics (loss, accuracy) don't surface geometric properties.
**Complexity:** High
**Implementation:**
- **Stiffness field visualization:** Plot `S = 2^(floor(log2|w|) - M)` for each weight, show how it changes over training
- **Quantization grid alignment:** Measure distance from actual weight values to nearest FP8 representable value (shows manifold discretization)
- **Local curvature estimation:** Hessian eigenvalue spectrum per layer (expensive, sample periodically)
- **Gradient-stiffness correlation:** Does stiffness-preconditioned gradient align with loss descent?

**Novel contribution:** No existing FP8 work measures optimizer-induced manifold geometry. This surfaces *why* exotic formats succeed/fail, not just *that* they do.

**Sources:**
- [Riemannian Optimization Survey (ACM 2025)](https://dl.acm.org/doi/10.1145/3708498) - Manifold optimization fundamentals
- [Neural Differential Manifold (ICLR 2025)](https://arxiv.org/abs/2510.25113) - Geometric structure in neural networks
- [Modular Manifolds Framework](https://thinkingmachines.ai/blog/modular-manifolds/) - Co-designing optimizers with manifold constraints

**Confidence:** MEDIUM (concept validated in Riemannian optimization, not yet applied to FP8 quantization specifically)

---

### 8. Bit-Space Uniformity Analysis
**What:** Treat FP8 as uniform in bit-space (not value-space) and measure whether updates respect this structure
**Why valuable:** E7M0 has 256 representable values (powers-of-2 only). Standard optimizers assume smooth value space; manifold-aware approach assumes uniform bit-space. This diagnostic tests the hypothesis.
**Complexity:** Medium-High
**Implementation:**
- **Bit-space delta tracking:** For each weight update, compute `Δbit = bit_index(w_new) - bit_index(w_old)` rather than `Δw = w_new - w_old`
- **Representability gap:** Measure how often desired update is unrepresentable in target FP8 format
- **Update success rate:** Fraction of updates that move weight to better loss region vs. forced to suboptimal representable neighbor
- **Format capacity utilization:** Which FP8 values are actually used? (E7M0 may only need subset of 256 values)

**Novel contribution:** Viewing quantization through bit-space lens is unique to this project. Standard work measures value-space metrics (overflow, dynamic range).

**Sources:**
- [Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Representable value grid structure
- [Bit-Level Scaling Laws (2025)](https://arxiv.org/html/2501.06218) - Bit-level analysis for vision models

**Confidence:** LOW-MEDIUM (novel framing, not directly validated in literature)

---

### 9. Format-Specific Stability Thresholds
**What:** Automatically detect stability limits for each FP8 format (max safe LR, batch size, etc.)
**Why valuable:** E7M0 (256 values) likely has radically different stability envelope than E4M3 (thousands of values). Automated detection saves manual tuning.
**Complexity:** High
**Implementation:**
- **Learning rate sweep:** For each format, binary search to find max stable LR (no divergence in first 1K steps)
- **Batch size sensitivity:** Test if larger batches stabilize extreme formats (E7M0)
- **Warmup requirement:** Does E7M0 need longer warmup than E4M3?
- **Format-specific divergence signatures:** Does E7M0 diverge via different patterns than E4M3? (sudden spike vs gradual drift)

**Novel contribution:** Treat each format as having unique training dynamics rather than trying to force universal hyperparameters.

**Sources:**
- [FP8 Training Stability (ICLR 2026)](https://arxiv.org/pdf/2511.05811) - Stability challenges in FP8
- [Transformer Divergence Detection](https://futureagi.com/blogs/evaluating-transformer-architectures-key-metrics-and-performance-benchmarks) - Conservative LR schedules for stability

**Confidence:** MEDIUM (concept of format-specific tuning is logical, not explicitly validated)

---

### 10. Comparative Dynamics Dashboard
**What:** Real-time visualization comparing all formats side-by-side during training
**Why valuable:** Budget constraint (≤$20) means can't run exhaustive sweeps. Need to see early indicators of which formats are working.
**Complexity:** Medium
**Implementation:**
- **Live parallel coordinates plot:** Each format as a line through axes (loss, grad norm, overflow rate, stiffness variance)
- **Divergence early warning:** Flag formats likely to diverge (kurtosis-based prediction from research)
- **Resource allocation hints:** "E1M6 and E4M3 stable, allocate remaining budget to E7M0 variants"
- **Interactive drill-down:** Click format → detailed diagnostics (weight histograms, gradient flows)

**Novel contribution:** Budget-aware experiment prioritization via early diagnostics.

**Sources:**
- [W&B Parallel Coordinates](https://docs.wandb.ai/tutorials/sweeps/) - Hyperparameter comparison visualization
- [FP8 Kurtosis Diagnostics](https://arxiv.org/html/2310.18313v2) - Predicting divergence from short runs

**Confidence:** MEDIUM (visualization tools exist, applying to format comparison is novel)

---

## Diagnostics & Metrics

Comprehensive list of what to measure and why, organized by diagnostic category.

### Numerical Health Metrics
| Metric | Frequency | Purpose | Critical Threshold |
|--------|-----------|---------|-------------------|
| Overflow count (activations) | Per step | Detect E4M3 range violations | >1% of values → likely divergence |
| Underflow count (gradients) | Per step | Detect gradient vanishing | >10% of gradients → dead neurons |
| NaN/Inf count | Per step | Catastrophic failure detection | Any occurrence → halt training |
| Dynamic range (amax tracking) | Per 100 steps | Verify scaling factors adapt | Amax growing exponentially → instability |
| Loss scale adjustment count | Per epoch | Monitor auto-scaling effectiveness | Frequent reductions → poor scaling strategy |

**Source confidence:** HIGH (derived from [NVIDIA Transformer Engine docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html), [FP8 auto-scaling research](https://www.emergentmind.com/topics/fp8-mixed-precision-training-framework))

---

### Convergence Metrics
| Metric | Frequency | Purpose | Success Criteria |
|--------|-----------|---------|------------------|
| Train loss | Per step | Primary objective | Monotonic decrease (smoothed) |
| Validation loss | Per epoch | Generalization check | Within 10% of BF16 baseline |
| Validation perplexity | Per epoch | Language modeling quality | Lower is better, compare to baseline |
| Loss gap (FP8 - BF16) | Per epoch | Quantization degradation | <5% gap = acceptable FP8 quality |
| Steps to convergence | End of run | Training efficiency | Fewer steps = better format |

**Source confidence:** HIGH (standard metrics from [Neptune AI training report](https://neptune.ai/state-of-foundation-model-training-report))

---

### Gradient Flow Metrics
| Metric | Frequency | Purpose | Warning Signs |
|--------|-----------|---------|---------------|
| Gradient L2 norm (per layer) | Per step | Detect vanishing/explosion | >10x variance across layers |
| Gradient SNR (FP8 vs FP32) | Per 100 steps | Quantization noise impact | SNR <10 dB → high noise |
| Dead neuron fraction | Per epoch | Detect optimization failure | >20% dead → poor initialization |
| Gradient clipping events | Per step | Monitor extreme gradients | >50% clipped → LR too high |
| Layer-wise gradient ratio | Per step | Balance depth of learning | Top/bottom layer ratio >100 → poor flow |

**Source confidence:** HIGH (from [FP8-LM gradient analysis](https://arxiv.org/pdf/2310.18313), [numerical stability fundamentals](http://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html))

---

### Manifold Geometry Metrics (Novel)
| Metric | Frequency | Purpose | Interpretation |
|--------|-----------|---------|----------------|
| Stiffness field entropy | Per 100 steps | Measure manifold smoothness | High entropy → rough quantization surface |
| Quantization misalignment | Per step | Distance to nearest FP8 value | High misalignment → poor format fit |
| Hessian top eigenvalue | Per 500 steps (expensive) | Local curvature | Eigenvalue >1000 → sharp minima |
| Bit-space update variance | Per step | Uniformity of bit-level steps | Low variance → structured exploration |
| Format capacity usage | Per epoch | Which FP8 values used | E7M0 using <50% → overparameterized |

**Source confidence:** MEDIUM (derived from [Riemannian optimization theory](https://dl.acm.org/doi/10.1145/3708498), [manifold learning](https://thinkingmachines.ai/blog/modular-manifolds/), not directly validated for FP8)

---

### Resource & Efficiency Metrics
| Metric | Frequency | Purpose | Budget Constraint |
|--------|-----------|---------|-------------------|
| Throughput (tokens/sec) | Per 100 steps | Training speed | Higher throughput = more experiments |
| Memory usage (GB) | Per epoch | Verify FP8 savings | FP8 should use ~50% of BF16 |
| Checkpoint size (MB) | Per checkpoint | Storage efficiency | Quantized checkpoints smaller |
| Wall-clock time to target loss | End of run | Total experiment cost | Critical for ≤$20 budget |
| GPU utilization (%) | Continuous | Hardware efficiency | >80% = good utilization |

**Source confidence:** HIGH (standard profiling metrics, FP8 memory savings from [TensorRT documentation](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html))

---

## Anti-Features

Features to deliberately NOT build. Maintaining diagnostic clarity is critical for ≤$20 budget.

### 1. Distributed Training Support
**What NOT to build:** Multi-GPU, data parallel, model parallel infrastructure
**Why avoid:** Single H100 constraint. Distributed training introduces synchronization overhead, gradient all-reduce quantization issues, and complexity that obscures core research question.
**What to do instead:** Optimize single-GPU utilization. Use gradient accumulation for effective larger batch sizes if needed.

**Source:** [FP8 Gradient All-Reduce Issues](https://www.emergentmind.com/topics/fp8-mixed-precision-training-framework) - Distributed FP8 introduces underflow in pre-scaling

**Confidence:** HIGH (explicit constraint from project context)

---

### 2. Production Optimizations
**What NOT to build:** Inference optimization, export to ONNX/TensorRT, deployment pipelines, model serving
**Why avoid:** This is a research test bench, not a deployment framework. Production features distract from experimental clarity and consume budget on irrelevant infrastructure.
**What to do instead:** Focus on training-time diagnostics. If inference needed, use simple eager-mode evaluation.

**Confidence:** HIGH (clear scope boundary)

---

### 3. Automatic Hyperparameter Tuning
**What NOT to build:** Bayesian optimization, AutoML, extensive grid searches across all hyperparameters
**Why avoid:** Budget ≤$20 means ~10-20 runs total. Automated tuning requires hundreds of trials. Manual, hypothesis-driven experiments are more informative.
**What to do instead:** Use literature-based sensible defaults (e.g., LR from nanoGPT). Reserve budget for format comparisons, not LR tuning.

**Source:** [W&B Sweeps](https://docs.wandb.ai/models/sweeps) - Bayesian sweeps require many trials to converge

**Confidence:** HIGH (budget constraint forces prioritization)

---

### 4. Comprehensive Baseline Suite
**What NOT to build:** Comparisons to dozens of optimizers (Adam, AdamW, SGD, LAMB, etc.) or architectures (GPT, BERT, T5, etc.)
**Why avoid:** Scope creep. Core question is "Do exotic FP8 formats work with manifold-aware optimization?" Not "What's the best optimizer?"
**What to do instead:** Single baseline: AdamW + BF16 (nanoGPT default). Compare manifold-aware approach across FP8 formats to this one baseline.

**Confidence:** HIGH (focused research scope)

---

### 5. Data Preprocessing Pipelines
**What NOT to build:** Custom tokenizers, data augmentation, curriculum learning, dataset balancing
**Why avoid:** Introduces confounds. Need to isolate FP8 format effects, not data effects.
**What to do instead:** Use pre-tokenized EurLex (legal documents) as-is. Fixed random seed for reproducibility.

**Confidence:** HIGH (experimental control)

---

### 6. Advanced Visualization Dashboards
**What NOT to build:** Real-time 3D manifold renderings, interactive loss landscapes, animated training trajectories
**Why avoid:** Nice-to-have, but diagnostically redundant with simpler plots. Consumes development time without proportional insight.
**What to do instead:** Static plots in W&B/TensorBoard are sufficient. Parallel coordinates for format comparison (simple, informative).

**Source:** [W&B Visualization](https://docs.wandb.ai/tutorials/sweeps/) - Simple plots often more interpretable than complex dashboards

**Confidence:** MEDIUM-HIGH (pragmatic tradeoff)

---

### 7. Extensive Stability Mitigations
**What NOT to build:** Per-layer scaling, architectural modifications (Smooth-SwiGLU), outlier suppression techniques
**Why avoid:** These techniques *hide* the problems exotic formats have. Research goal is to see if manifold-aware optimizer *solves* instability, not if architecture hacks do.
**What to do instead:** Minimal baseline (standard Transformer, no special tricks). Let FP8 format + optimizer combination succeed or fail on its own.

**Source:** [FP8 Stability Techniques](https://arxiv.org/pdf/2310.18313) - Architectural tricks improve stability but obscure fundamental format limitations

**Confidence:** HIGH (research integrity—don't hide the phenomena you're studying)

---

## Feature Dependencies

Understanding what must be built before what.

```
Foundation Layer (build first):
├─ Checkpoint Management (enables recovery)
├─ Training Loss Tracking (validates basic functionality)
└─ Format Ablation Support (core experimental capability)

Diagnostic Layer (build second):
├─ Numerical Stability Monitoring (depends on: Format Ablation)
├─ Gradient Statistics (depends on: Training Loop)
├─ Weight Distribution Tracking (depends on: Checkpoint Management for periodic sampling)
└─ Resource Metrics (depends on: Training Loop)

Advanced Diagnostics (build third):
├─ Manifold Geometry Diagnostics (depends on: Weight Distribution, Gradient Statistics)
│   └─ Stiffness field visualization requires weight values
│   └─ Curvature estimation requires gradient computations
├─ Bit-Space Uniformity Analysis (depends on: Weight Distribution, Numerical Stability)
│   └─ Representability gap requires knowing FP8 format spec
└─ Format-Specific Stability Thresholds (depends on: Numerical Stability, Convergence Metrics)

Visualization Layer (build last):
└─ Comparative Dynamics Dashboard (depends on: All Diagnostics)
    └─ Aggregates metrics from all prior layers
```

**Critical path for MVP:**
1. Format Ablation Support → Can run experiments
2. Training Loss Tracking → Know if it works
3. Numerical Stability Monitoring → Detect failure modes
4. Gradient Statistics → Debug optimization issues
5. Manifold Geometry Diagnostics → Test core hypothesis

**Defer to post-MVP:**
- Bit-Space Uniformity Analysis (nice-to-have, not critical for initial validation)
- Format-Specific Stability Thresholds (can manually tune initially)
- Comparative Dynamics Dashboard (can use static plots from W&B)

---

## MVP Recommendation

For a ≤$20 budget test bench, prioritize **diagnostic depth over breadth**. Better to deeply understand why E7M0 fails than to shallowly compare 10 formats.

### Must-Have (MVP Phase 1)
1. **Format Ablation Support** - Test E3M4, E4M3, E5M2, E7M0 (skip E1M6 initially—too many mantissa bits, less interesting)
2. **Numerical Stability Monitoring** - Overflow/underflow rates, NaN detection
3. **Training Loss & Metrics** - Basic convergence validation
4. **Gradient Statistics** - Detect vanishing/explosion
5. **Checkpoint Management** - Survive spot instance interruptions
6. **Manifold Geometry Diagnostics (minimal)** - Stiffness field visualization only (defer Hessian eigenvalues)

**Estimated cost:** ~$10-12 (leaves budget for follow-up experiments)

### Should-Have (MVP Phase 2, if budget remains)
7. **Weight Distribution Tracking** - See how weights cluster in FP8 space
8. **Bit-Space Uniformity Analysis (partial)** - Representability gap metric only
9. **Comparative Dynamics Dashboard (simple)** - Parallel coordinates plot in W&B

**Estimated cost:** ~$5-7

### Won't-Have (Post-MVP)
- Format-Specific Stability Thresholds (manual tuning acceptable for initial paper)
- Advanced curvature estimation (Hessian eigenvalues expensive, defer until hypothesis validated)
- Real-time dashboards (static plots sufficient)

---

## Open Questions & Research Gaps

Areas where literature is sparse or contradictory, flagged for future investigation.

### 1. E7M0 Viability
**Question:** Can E7M0 (256 powers-of-2) ever train a transformer, or is dynamic range too limited?
**Literature gap:** No published work on E7M0 for neural networks. E5M2 is most extreme tested format.
**Implication:** E7M0 experiment is high-risk, high-reward. May need fallback to E5M2 if fundamentally impossible.
**Confidence:** LOW (no prior art)

### 2. Stiffness Preconditioning Optimality
**Question:** Is `S = 2^(floor(log2|w|) - M)` the right stiffness formula, or should exponent bits E also factor in?
**Literature gap:** Manifold-aware optimization for quantization is novel—no existing validation.
**Implication:** May need to test alternative stiffness formulas (ablation within ablation).
**Confidence:** LOW (theoretical derivation, not empirically validated)

### 3. Bit-Space vs Value-Space Optimization
**Question:** Does uniform bit-space really matter, or is it a mathematical curiosity?
**Literature gap:** Quantization research measures value-space metrics. Bit-space framing is novel.
**Implication:** Bit-space uniformity analysis is speculative—may not correlate with performance.
**Confidence:** LOW (untested hypothesis)

### 4. Format-Dependent Learning Rates
**Question:** Do extreme formats (E7M0) need radically different LRs, or is AdamW adaptive enough?
**Literature gap:** LR tuning research assumes FP16/BF16. FP8 work uses standard schedules with outlier suppression.
**Implication:** May discover E7M0 needs 10x lower LR, making it impractical.
**Confidence:** MEDIUM (logical concern, not directly studied)

### 5. EurLex Suitability
**Question:** Are legal documents (long, formal) harder/easier for FP8 than general text?
**Literature gap:** FP8 research uses C4, Pile, WikiText. Domain-specific effects unknown.
**Implication:** Results may not generalize to other domains. Consider fallback to WikiText if EurLex shows anomalous behavior.
**Confidence:** MEDIUM (domain effects plausible but unstudied)

---

## Sources Summary

**HIGH Confidence (Context7 / Official Docs):**
- NVIDIA Transformer Engine documentation (FP8 format specs)
- PyTorch documentation (checkpointing, mixed precision)
- ICLR/AAAI/ACL 2025-2026 papers (peer-reviewed FP8 research)

**MEDIUM Confidence (Multiple Credible Sources):**
- Riemannian optimization surveys (ACM, arXiv)
- W&B/Neptune documentation (experiment tracking best practices)
- Visual guides from recognized ML educators

**LOW Confidence (Single Source / Novel Hypotheses):**
- Bit-space uniformity framing (project-specific, not validated)
- E7M0 format viability (no prior art)
- Stiffness preconditioning formula (theoretical, not empirical)

---

## Conclusion

This test bench should be **diagnostic-first, not feature-rich**. The ≤$20 budget and single-H100 constraint force ruthless prioritization: measure what matters (manifold geometry, numerical stability), skip what doesn't (distributed training, AutoML).

**Core differentiator:** Treating FP8 quantization as a manifold geometry problem, not just a numerical precision problem. This requires novel diagnostics (stiffness field visualization, bit-space uniformity) not found in existing FP8 frameworks.

**Biggest risk:** E7M0 may be fundamentally untrainable, making half the experiments fail. Mitigation: Front-load E7M0 experiments (fail fast), have E5M2 as backup to still demonstrate manifold-aware approach.

**Success metric:** Not "does E7M0 match BF16 accuracy" but "do we understand *why* E7M0 fails differently than E4M3, and does manifold-aware optimization change the failure mode?" Diagnostic depth > absolute performance.
