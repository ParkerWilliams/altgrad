# Project Research Summary

**Project:** AltGrad - FP8 Manifold-Aware Optimizer Test Bench
**Domain:** Low-precision transformer training with custom floating-point formats
**Researched:** 2026-01-20
**Confidence:** MEDIUM-HIGH

## Executive Summary

AltGrad is a research platform for testing exotic FP8 formats (E1M6, E3M4, E5M2, E7M0) with manifold-aware optimization. Based on extensive 2025-2026 research, the recommended approach is **simulated quantization** using PyTorch 2.8's native FP8 types for baseline formats (E4M3/E5M2), with custom quantize/dequantize operations for exotic formats. The architecture follows a proven pattern: fake quantization with Straight-Through Estimators (STE) at layer boundaries, wrapped around a nanoGPT backbone. Critical to success is treating FP8 quantization as a geometric manifold problem rather than just numerical precision degradation.

The key risk is **outlier explosion leading to immediate divergence**, which affects all FP8 training but is catastrophic for extreme formats like E7M0. Mitigation requires comprehensive monitoring (kurtosis tracking, per-tensor scaling, gradient statistics) established BEFORE custom format experiments. E7M0 specifically has a >90% probability of training failure due to zero-gradient regions created by powers-of-2 quantization—this should be treated as a negative result experiment to document failure modes, not production-ready training.

The test bench must prioritize diagnostic depth over feature breadth. With a ≤$20 budget on single H100 GPU, success means understanding WHY E7M0 fails differently than E4M3, not achieving BF16-equivalent accuracy. The recommended stack (PyTorch 2.8 + TorchAO + custom ops + Geoopt + nanoGPT) enables rapid format experimentation while maintaining scientific rigor.

## Key Findings

### Recommended Stack

The modern FP8 training ecosystem in 2026 centers on PyTorch's native support for E4M3/E5M2 formats, with custom operations required for exotic variants. Start with TorchAO for baseline validation, then implement custom quantization using `torch.library.custom_op` for E1M6/E3M4/E7M0.

**Core technologies:**
- **PyTorch 2.8+**: Native float8 types (E4M3fn, E5M2), custom_op API for exotic formats, excellent H100 FP8 Tensor Core support
- **TorchAO 0.10+**: One-line FP8 conversion for baseline experiments, proven at 2k GPU scale, but limited to standard E4M3/E5M2
- **Geoopt 0.5.1**: Riemannian optimization framework requiring custom FP8Manifold implementation for manifold-aware optimizer
- **NanoGPT**: Simple, hackable transformer backbone proven compatible with FP8 (modded-nanoGPT demonstrates custom ops on H100)
- **NNGeometry 0.4**: Fisher Information Matrix computation for natural gradient methods (KFAC/EKFAC approximations)

**Critical version requirements:** PyTorch 2.8+ (native FP8), CUDA 12.6+ (Tensor Core support), Python 3.12 (NVIDIA recommendation)

**What to avoid:** QPyTorch (software-only simulation, no H100 acceleration), Transformer Engine (only supports E4M3/E5M2, not flexible enough for custom formats), distributed training frameworks (single GPU constraint)

### Expected Features

The test bench is diagnostic-first, not feature-rich. Budget constraints force ruthless prioritization toward manifold geometry insights rather than production optimizations.

**Must have (table stakes):**
- **Numerical stability monitoring**: Per-tensor overflow/underflow tracking, kurtosis-based divergence prediction, NaN/Inf detection with early stopping
- **Training loss & metrics**: Train/val loss, perplexity, wall-clock time, comparison plots vs BF16 baseline
- **Gradient statistics**: Per-layer norms, SNR (FP8 vs FP32), dead neuron detection, clipping event monitoring
- **Checkpoint management**: Periodic saves with restart capability (critical for spot instance risk under ≤$20 budget)
- **Format ablation support**: Train identical model with different FP8 formats, controlling all variables except mantissa/exponent split
- **Weight distribution tracking**: Histograms showing clustering in representable regions vs quantization boundaries

**Should have (competitive differentiators):**
- **Manifold geometry diagnostics**: Stiffness field visualization (S = 2^(floor(log2|w|) - M)), quantization grid alignment, gradient-stiffness correlation
- **Bit-space uniformity analysis**: Treat FP8 as uniform in bit-space (not value-space), measure representability gaps and update success rates
- **Format-specific stability thresholds**: Automated detection of max safe learning rate per format (E7M0 likely needs radically different LR)
- **Comparative dynamics dashboard**: Real-time parallel coordinates plot comparing all formats, early divergence warning

**Defer (anti-features to avoid):**
- Distributed training support (single GPU constraint)
- Production optimizations (inference, ONNX export, model serving)
- Automatic hyperparameter tuning (budget too limited for Bayesian optimization)
- Comprehensive baseline suite (focus on single AdamW+BF16 baseline)
- Data preprocessing pipelines (use EurLex as-is to avoid confounds)
- Advanced 3D visualizations (static plots sufficient, don't consume dev time)

### Architecture Approach

FP8 test benches follow a **simulated quantization architecture** where tensors are stored in high precision (FP32/BF16) but quantized at specific boundaries to emulate low-precision behavior. The pattern is: quantize (FP32→FP8 sim) → dequantize (FP8 sim→FP32) → compute (FP32), with Straight-Through Estimators enabling gradient flow. This "quantization sandwich" preserves PyTorch compatibility while simulating numerical effects.

**Major components:**
1. **Quantization Engine** (independent core): Format registry (E1M6/E3M4/E5M2/E7M0 specs), scaling factor calculator (per-tensor dynamic/static), quantize/dequantize functions with STE gradient override
2. **Fake Quantization Wrapper Layers**: Inject quantization at layer boundaries without modifying nanoGPT core—QuantizedLinear wraps nn.Linear with format-specific quantize/dequantize
3. **Training Loop Integration**: Strategic quantization at 6 points (QKV projections, attention output, MLP FC1/FC2, classifier head, gradient tensors), supporting per-layer mixed precision
4. **NanoGPT Integration via Model Surgery**: Replace Linear layers post-initialization with wrappers (no forking), preserving original model.py unchanged
5. **Custom Optimizer with Stiffness**: Extend AdamW to inject stiffness factor (grad *= stiffness_factor) before parameter update, or use wrapper pattern for dynamic schedules
6. **Bit-Position Tracking System**: Monitor which mantissa/exponent bits are active, track bit-level statistics, detect overflow/underflow events, log format utilization to WandB

**Key patterns to follow:**
- **Separation of concerns**: Quantization logic independent of model definition (enables easy format experimentation)
- **Lazy scale computation**: Weight scales computed once (static), activation scales per-batch (dynamic)
- **Format polymorphism**: All FP8 formats share same interface (quantize, dequantize, compute_scale)
- **Instrumentation without overhead**: Tracking hooks disabled via config flag when not needed

**Critical architectural decisions:**
- Keep attention and LayerNorm in BF16 (numerically sensitive, small compute fraction in ≤50M params)
- Use per-tensor scaling with delayed amax tracking (history buffer length 16-32 for smooth updates)
- Quantize gradients with E5M2 (higher dynamic range than E4M3)
- Store optimizer state (momentum, variance) in FP32 (second-order methods need precision)

### Critical Pitfalls

1. **Extreme outlier explosion → immediate divergence**: Unmitigated FP8 training causes activations to reach 1000× std dev, exceeding E4M3 range (±448). Kurtosis spikes precede divergence. Prevention: Monitor kurtosis continuously (alert when >10), per-tensor scaling with amax tracking, outlier suppression at residual outputs, selective BF16 fallback for LayerNorm/Softmax. **E7M0 risk: Zero mantissa bits create extreme quantization error for bell-shaped distributions—immediate collapse highly likely.**

2. **Global loss scaling failure → gradient underflow/overflow**: Single scale factor infeasible for FP8 (E4M3 range: ±448 vs FP16: ±65,504). Prevention: Per-tensor scaling with unique factors, delayed scaling using amax history, FP32 scaling factors, gradient accumulation with re-scale before accumulation. **E7M0 risk: Powers-of-2 quantization with global scaling creates systematic gradient bias.**

3. **E7M0 powers-of-2: Zero gradient regions**: No mantissa bits → only {..., 0.5, 1.0, 2.0, 4.0, ...} representable. Values in interval (2^k, 2^(k+1)) collapse to boundaries, creating zero-gradient zones where small perturbations produce no output change. **Failure probability >90% for standard transformer training.** Prevention: Treat as negative result experiment, document failure modes (gradient sparsity, update sparsity, training plateau within 10 steps).

4. **Stochastic rounding variance explosion → non-convergence**: SR provides unbiased estimates but introduces variance that prevents convergence when batch too small or precision too low. Prevention: Split rounding strategy (SR only in backward, round-to-nearest in forward), minimum batch size to average SR variance, gradient clipping before quantization. **E7M0 risk: Massive quantization intervals (gap from 2^k to 2^(k+1)) create enormous SR variance.**

5. **Manifold-aware optimizer numerical instability**: Second-order methods (Hessian estimation, curvature-based preconditioning) amplify quantization errors. Prevention: FP32 preconditioner state, diagonal approximations instead of full Hessian, damping (add λI before inversion), periodic resets to prevent error accumulation, verify metric tensor stays positive definite in FP8.

## Implications for Roadmap

Based on research, suggested phase structure prioritizes independent component development, early validation of baseline formats, and treating E7M0 as high-risk negative result experiment:

### Phase 1: Quantization Engine (Independent Foundation)
**Rationale:** Build and validate quantization mechanics in isolation before any training integration. This enables parallel development and thorough testing of novel formats without NanoGPT complexity.
**Delivers:** Standalone quantization.py module with FP8Format class (E4M3, E5M2, E1M6, E3M4, E7M0), STE autograd function, unit tests validating round-trip properties and gradient flow.
**Addresses:** Infrastructure for format ablation support (table stakes), foundation for bit-space uniformity analysis (differentiator).
**Avoids:** Custom format implementation bugs (pitfall #11) through property-based testing, boundary value validation, bit-level inspection.

### Phase 2: Baseline Validation (BF16 + Standard FP8)
**Rationale:** Establish stable BF16 baseline and validate E4M3/E5M2 implementations before custom format experiments. Critical for detecting E5M2 maxval bug and establishing monitoring infrastructure.
**Delivers:** NanoGPT training on EurLex in BF16 (baseline metrics), TorchAO-based E4M3/E5M2 training (standard FP8 validation), comprehensive monitoring system (kurtosis, overflow, gradient stats), H100 Tensor Core utilization profiling.
**Uses:** PyTorch 2.8, TorchAO, NanoGPT, Hugging Face Datasets (EurLex).
**Addresses:** Numerical stability monitoring, training loss tracking, gradient statistics, checkpoint management (all table stakes).
**Avoids:** Outlier explosion (pitfall #1) via kurtosis monitoring, E5M2 maxval bug (pitfall #3) via implementation verification, attention entropy collapse (pitfall #6) via BF16 attention.

### Phase 3: Wrapper Layer Integration (Model Surgery)
**Rationale:** Inject custom quantization into NanoGPT without forking, enabling per-layer mixed precision experiments.
**Delivers:** QuantizedLinear wrapper layers, quantize_model() surgery function, integrated training loop with custom optimizer (StiffnessAdamW or wrapper pattern), bit-tracking system integrated with WandB logging.
**Implements:** Fake quantization wrappers (architecture component #2), training loop integration (component #3), NanoGPT integration (component #4).
**Addresses:** Format ablation support with per-layer granularity, bit-space uniformity analysis infrastructure.
**Avoids:** Forking NanoGPT (anti-pattern #1), quantizing embeddings (anti-pattern #2), hardcoding format specs (anti-pattern #5).

### Phase 4: Custom Format Testing (E1M6, E3M4, Baseline E7M0)
**Rationale:** Test exotic formats systematically, treating E7M0 as negative result experiment to document failure modes.
**Delivers:** E3M4 convergence validation (likely viable), E1M6 testing on normalized tensors only (post-LayerNorm activations), E7M0 negative result documentation (gradient sparsity, zero-gradient regions, training collapse timeline).
**Addresses:** Format-specific stability thresholds (differentiator), manifold geometry diagnostics for formats that converge.
**Avoids:** E7M0 zero gradient regions (pitfall #3) by expecting failure and documenting scientifically, E1M6 narrow range overflow (pitfall #10) by restricting to normalized tensors, stochastic rounding variance (pitfall #4) via split rounding strategy.

### Phase 5: Manifold-Aware Optimizer Integration
**Rationale:** Combine custom formats with novel optimizer only after formats proven viable independently. Isolate optimizer effects from format effects.
**Delivers:** Custom FP8Manifold class in Geoopt framework, stiffness-based preconditioner (S = 2^(floor(log2|w|) - M)), Riemannian optimizer integration with E3M4 (if viable from Phase 4), constraint violation monitoring (orthogonality, positive-definiteness).
**Uses:** Geoopt, NNGeometry (Fisher computation), custom manifold implementation.
**Addresses:** Manifold geometry diagnostics (stiffness field visualization, curvature estimation), core research hypothesis testing.
**Avoids:** Geometric structure corruption (pitfall #13) via FP32 preconditioner state and periodic re-orthogonalization, stiffness preconditioning instability (pitfall #12) via damping and diagonal approximations.

### Phase 6: Comparative Analysis & Documentation
**Rationale:** Synthesize findings across formats, produce scientific documentation of results.
**Delivers:** Comparative dynamics dashboard (parallel coordinates plot), format-by-format convergence analysis, roadmap implications summary, publication-ready figures and tables.
**Addresses:** Comparative dynamics dashboard (differentiator), critical data size detection (pitfall #5).

### Phase Ordering Rationale

- **Quantization Engine first** enables parallel development of formats while NanoGPT integration proceeds independently—no blocking dependencies.
- **Baseline validation before custom formats** prevents confusing implementation bugs with fundamental format limitations (E5M2 maxval bug must be caught early).
- **Wrapper integration before format testing** establishes per-layer mixed precision infrastructure needed for E1M6 (restricted to normalized tensors).
- **Custom formats before manifold optimizer** isolates format viability from optimizer complexity—if E7M0 fails with standard SGD, it will definitely fail with manifold-aware optimization.
- **Manifold optimizer last** leverages all prior diagnostics and validated formats, reducing risk of compounding unknowns.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (E7M0 testing):** No literature on E7M0 viability for neural networks—expect to pioneer negative results. May need `/gsd:research-phase` for E7M0-specific mitigation strategies (though likely futile).
- **Phase 5 (Manifold optimizer):** Custom FP8Manifold implementation has no existing library support—will require differential geometry theory consultation and empirical validation.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Quantization Engine):** Well-documented quantization mechanics, STE implementation has research paper (Yin et al., ICLR 2019) and PyTorch examples.
- **Phase 2 (Baseline Validation):** Standard FP8 training patterns extensively documented by NVIDIA, TorchAO, and 2025-2026 research literature.
- **Phase 3 (Wrapper Integration):** Model surgery pattern proven in modded-nanoGPT, PyTorch QAT tutorials provide reference implementations.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | PyTorch 2.8 + TorchAO extensively documented, modded-nanoGPT proves viability for custom ops on H100 |
| Features | MEDIUM-HIGH | Table stakes validated across multiple authoritative sources, differentiators are novel (manifold diagnostics) but theoretically grounded |
| Architecture | HIGH | Simulated quantization with STE is standard QAT pattern, NVIDIA Transformer Engine and PyTorch provide reference architectures |
| Pitfalls | HIGH | Standard FP8 pitfalls from extensive 2025-2026 research, E7M0 predictions based on theory + analogous research |

**Overall confidence:** MEDIUM-HIGH

Research is strong for baseline FP8 (E4M3/E5M2) and standard pitfalls. Confidence drops to MEDIUM for novel aspects:
- E7M0 viability predictions (no prior art, >90% failure probability is theoretical)
- Manifold-aware optimizer integration (novel combination, limited literature on second-order methods + extreme quantization)
- Bit-space uniformity analysis (project-specific framing, not validated in literature)

### Gaps to Address

Areas requiring validation during implementation:

- **E7M0 failure timeline**: Literature predicts immediate collapse, but exact failure mode (zero gradients vs exploding gradients vs update sparsity) requires empirical observation. **Mitigation:** Instrument Phase 4 with extensive gradient statistics to document precise failure mechanism.

- **Stiffness preconditioning formula**: `S = 2^(floor(log2|w|) - M)` is theoretical derivation, not empirically validated. Alternative formulas incorporating exponent bits may perform better. **Mitigation:** Treat Phase 5 as ablation study, test multiple stiffness formulas.

- **E1M6 viable use cases**: Hypothesis is post-LayerNorm activations, but which layers specifically? **Mitigation:** Phase 4 should profile dynamic range per layer before restricting E1M6 usage.

- **Batch size sweet spot**: FP8 speedup requires sufficient arithmetic intensity, but small models may be memory-bandwidth bound. Budget constraints limit batch size exploration. **Mitigation:** Phase 2 must profile performance across batch sizes [8, 16, 32, 64] to establish expectations.

- **EurLex domain effects**: Legal documents (long, formal) may behave differently under quantization than general text. **Mitigation:** If Phase 2 shows anomalous behavior, have WikiText as fallback dataset.

- **Manifold constraint violations**: How much drift from geometric constraints is acceptable before optimizer fails? **Mitigation:** Phase 5 must establish violation budgets empirically (e.g., ||Q^T Q - I|| < threshold).

## Sources

### Primary (HIGH confidence)
- PyTorch 2.8 Release Notes & Documentation (native FP8 types, custom_op API)
- NVIDIA Transformer Engine Documentation (FP8 primer, scaling strategies, H100 Tensor Core specs)
- TorchAO GitHub + PyTorch Blog (production FP8 training at scale, MXFP8 performance)
- ICLR/ACL/AAAI 2025-2026 Papers (peer-reviewed FP8 research, stability analysis, quantization techniques)
- Geoopt Documentation (Riemannian optimization API, manifold operations)
- modded-nanoGPT GitHub (proven H100 FP8 custom ops implementation)

### Secondary (MEDIUM confidence)
- ACM Computing Surveys (Riemannian optimization survey, geometric deep learning)
- Neptune AI / W&B Documentation (experiment tracking best practices, visualization patterns)
- Visual guides and educational content (quantization mechanics, bit-space analysis)
- Multiple research papers on specific topics (KFAC, natural gradients, stochastic rounding)

### Tertiary (LOW confidence, requires validation)
- Bit-space uniformity framing (project-specific hypothesis, not validated in literature)
- E7M0 format viability predictions (no prior art, theoretical extrapolation)
- Stiffness preconditioning formula (theoretical derivation, needs empirical validation)
- Custom FP8Manifold implementation approach (no existing library, novel contribution)

---
*Research completed: 2026-01-20*
*Ready for roadmap: yes*
