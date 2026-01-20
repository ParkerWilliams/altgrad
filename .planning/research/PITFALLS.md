# Pitfalls Research: FP8 Quantized Training

**Domain:** Low-precision transformer training with custom FP8 formats
**Researched:** 2026-01-20
**Confidence:** HIGH (based on extensive 2025-2026 research literature)
**Context:** Single H100, ≤50M params, custom E7M0/E1M6 formats alongside standard E4M3/E5M2

---

## Critical Failures (Will Definitely Break Training)

### 1. Extreme Outlier Explosion → Immediate Divergence

**What goes wrong:**
Unmitigated FP8 training causes numerical instability due to extreme outlier values in Transformer residual streams, which exceed the representable range of FP8 (±448 for E4M3), resulting in overflows and catastrophic divergence. Activations can reach magnitudes **1000× the standard deviation** of the distribution, especially in:
- Residual streams after attention and FFN blocks
- SwiGLU activations (quadratic growth mechanism)
- QKV projections in attention layers

**Why it happens:**
- **Architectural artifacts**: SwiGLU creates quadratic activation growth through element-wise multiplication
- **Weight colinearity**: Progressive alignment of weight matrices during training mechanically drives larger activations
- **Cascading effect**: Outliers in early layers amplify through deeper layers

**Warning signs:**
- Kurtosis spikes in activation distributions (average kurtosis >10 in QKV or block outputs)
- Activation max values exceeding FP8 range (>448 for E4M3, >57344 for E5M2)
- Gradient norm explosions (>100× baseline)
- Loss oscillations or sudden jumps (>2× step-over-step)
- NaN/Inf in attention scores or FFN outputs

**Prevention:**
1. **Monitor kurtosis continuously**: Track activation kurtosis at every residual block output (precedes observable divergence by substantial margins)
2. **Per-tensor scaling with amax tracking**: Use history buffer (length 16-32) for smooth scale updates
3. **Outlier suppression at residual outputs**: Apply clamping or smooth activation functions
4. **Selective BF16 fallback**: Keep LayerNorm, Softmax, final projection in BF16
5. **Architecture modifications**: Consider Smooth-SwiGLU variant to reduce outlier amplification

**Your E7M0 risk:**
Powers-of-2 only format (E7M0) has **zero mantissa bits** → cannot represent values between powers of 2 → **EXTREME quantization error** for typical neural network distributions which are bell-shaped around zero. **Immediate training collapse is highly likely.** Even activations in range [0.5, 1.0) will round to 0.5 or 1.0, losing all intermediate information.

**Phase mapping:** Phase 1 (Baseline validation) - Must establish monitoring BEFORE E7M0 experiments

---

### 2. Global Loss Scaling Failure → Gradient Underflow/Overflow

**What goes wrong:**
The single loss scaling factor strategy that worked for FP16 is **infeasible for FP8**. While FP8's dynamic range can store any particular activation or gradient, it **cannot store all of them simultaneously** with one scale factor. This causes:
- Gradient underflow: Small gradients round to zero (learning stalls)
- Gradient overflow: Large gradients saturate to ±max (NaN explosion)
- Gradient communication errors during distributed training (FP8 all-reduce)

**Why it happens:**
FP8 has extremely limited dynamic range (E4M3: ±448, vs FP16: ±65,504). A scale factor that prevents overflow for large activations will underflow small gradients, and vice versa.

**Warning signs:**
- Training loss plateaus with non-zero gradients
- Gradient histograms showing clipping at zero
- NaN/Inf after gradient all-reduce operations
- Divergence after N gradient accumulation steps
- Validation loss not improving despite training loss decrease

**Prevention:**
1. **Per-tensor scaling**: Unique scaling factor per tensor (weights, activations, gradients)
2. **Delayed scaling**: Use maximum absolute values from previous iterations (amax_history_len=16)
3. **FP32 scaling factors**: Store scales in FP32 for precision alignment and stability
4. **Gradient accumulation awareness**: Re-scale before accumulation, not after (prevents underflow)
5. **MXFP8 for high variance tensors**: Block-wise scaling (32 values per scale) for problematic layers

**Your E7M0 risk:**
Powers-of-2 quantization with global scaling will cause **systematic bias** in gradients. Small gradient components will consistently round to wrong power of 2, creating biased updates. Stochastic rounding REQUIRED but insufficient.

**Phase mapping:** Phase 2 (Infrastructure) - Implement per-tensor scaling infrastructure before custom format testing

---

### 3. E5M2 Max Value Bug → Silent NaN Injection

**What goes wrong:**
The E5M2 format reserves its highest exponent bits for ±Inf and NaN, so the maximum representable finite value is **57,344** (not 98,304). Using 98,304 as the quantization ceiling causes massive errors: many tensor components become Inf/NaN during quantization. This is a **known implementation bug** that silently corrupts training.

**Why it happens:**
Confusion about FP8 spec: E4M3fn ("finite numbers") excludes Inf, but E5M2 includes ±Inf. Incorrect maxval functions return wrong ceilings.

**Warning signs:**
- Gradients suddenly become NaN after specific layers
- Inf values appearing in E5M2 tensors
- Training divergence only when using E5M2 for gradients
- Errors appearing during backward pass but not forward

**Prevention:**
1. **Verify maxval implementation**: E4M3fn max = 448, E5M2 max = 57,344
2. **Test quantization bounds**: Unit test that values near max quantize correctly
3. **NaN detection hooks**: Register forward/backward hooks to catch NaN injection points
4. **Gradient sanitization**: Check for Inf/NaN before optimizer step

**Your custom format risk:**
You'll be implementing quantization from scratch for E7M0/E1M6. **Triple-check representable ranges.** Powers-of-2 formats have discontinuous ranges (e.g., gaps between 2^k and 2^(k+1)).

**Phase mapping:** Phase 1 (Baseline) - Validate E4M3/E5M2 implementations before proceeding

---

### 4. Stochastic Rounding Variance Explosion → Non-Convergence

**What goes wrong:**
Stochastic rounding (SR) provides unbiased gradient estimates but introduces **variance** that can prevent convergence when:
- Batch size too small (variance dominates signal)
- Applied in forward pass (compounds across layers)
- Precision too low (quantization noise > gradient signal)

At a critical point, when full-precision gradient std dev falls below √3 × quantization noise std dev, **training stops making progress**.

**Why it happens:**
SR rounds probabilistically based on distance to nearest levels. In low precision, many gradients fall between quantization levels, creating high-variance updates. Small batches amplify this variance.

**Warning signs:**
- Loss oscillates without converging (variance-dominated updates)
- Gradient norm variance increases over time
- Different training runs with same hyperparams diverge significantly
- Weight oscillations prevent finding local minima
- Training converges better with larger batch size but not smaller LR

**Prevention:**
1. **Split rounding strategy**: SR only in backward pass, round-to-nearest in forward
2. **Minimum batch size**: Ensure batch size large enough to average out SR variance
3. **Precision transition**: Switch to higher precision (BF16) for final QAT finetuning
4. **Gradient clipping**: Clip per-sample gradients before SR quantization
5. **Variance monitoring**: Track gradient variance across batches

**Your E7M0 risk:**
Powers-of-2 quantization has **massive quantization intervals** (gap from 2^k to 2^(k+1) grows exponentially). SR variance will be enormous for values in mid-interval. May need deterministic rounding or compensated summation.

**Phase mapping:** Phase 3 (Format testing) - Establish SR baseline with E4M3 before E7M0

---

## Subtle Issues (Silent Degradation)

### 5. Critical Data Size Phenomenon → Inverse Scaling

**What goes wrong:**
Recent research discovered that low-precision training exhibits a **"critical data size"**: beyond this threshold, **more training data inversely degrades model performance**. This contradicts standard scaling laws and can cause mysterious quality regressions.

**Why it happens:**
Quantization errors accumulate differently at scale. Early layer errors propagate and amplify through deeper layers in large models trained on more data. The compounding effect eventually dominates learning signal.

**Warning signs:**
- Validation loss increases after certain training steps (despite training loss decreasing)
- Model performs worse when trained longer
- Metrics plateau then degrade
- Hyperparameters that worked at small scale fail at larger scale

**Detection:**
- Plot validation metrics vs training tokens (look for inflection point)
- Compare small-data and large-data runs at same step count
- Monitor per-layer quantization error accumulation

**Prevention:**
1. **Quantization-aware finetuning**: Train in BF16, then QAT for final X% of tokens
2. **Mixed-precision checkpointing**: Periodically switch to BF16 for critical phases
3. **Reduced quantization early**: Use higher precision for first N% of training
4. **Data curriculum**: Gradually increase data volume while monitoring for degradation

**Your scenario:**
With ≤50M params and limited budget, you're unlikely to hit critical data size. But track validation metrics carefully during E7M0 runs—extreme quantization may show this effect earlier.

**Phase mapping:** Phase 5 (Analysis) - Include in comparative analysis metrics

---

### 6. Attention Entropy Collapse → Training Instability

**What goes wrong:**
Low attention entropy (highly concentrated attention scores) correlates with training instability: oscillating loss or divergence. Quantization exacerbates this by reducing precision in softmax computations, causing premature entropy collapse.

**Why it happens:**
Attention scores naturally sharpen during training. Quantization errors in QK^T computation cause some scores to dominate prematurely, creating "entropy collapse" where attention focuses on few tokens, starving others of gradient signal.

**Warning signs:**
- Attention entropy <1.0 across multiple heads
- Repetition bias in outputs (same tokens repeated)
- Representation collapse (hidden states become nearly parallel)
- Training plateau followed by sudden degradation

**Detection:**
- Monitor per-head attention entropy: H = -Σ p(i) log p(i)
- Track output token diversity (n-gram entropy)
- Measure hidden state similarity (cosine similarity between tokens)

**Prevention:**
1. **Keep attention in BF16**: QK^T and Softmax are numerically sensitive
2. **Spectral normalization**: σReparam reparametrizes linear layers to prevent pathological sharpening
3. **Entropy regularization**: Add penalty term for low attention entropy
4. **Early detection thresholds**: Alert when entropy drops below architecture-specific threshold

**Your scenario:**
Attention is small fraction of compute in ≤50M models, keeping it in BF16 has minimal overhead. **Strongly recommend BF16 attention for all formats.**

**Phase mapping:** Phase 1 (Baseline) - Establish entropy monitoring before format experiments

---

### 7. Exponent/Mantissa Balance Mismatch → Suboptimal Accuracy

**What goes wrong:**
The optimal split between exponent and mantissa bits depends on your data distribution. Research shows **exponent bits contribute slightly more than mantissa bits**, but the optimal balance varies. Choosing wrong split (e.g., E1M6 vs E3M4 for 8-bit) can degrade accuracy without obvious symptoms.

**Why it happens:**
Neural network distributions are non-uniform (bell-shaped, heavy-tailed). Exponent bits provide dynamic range (handle outliers), mantissa bits provide precision (handle common values). Mismatch between format and distribution causes either:
- Lost outliers (insufficient exponent range)
- Lost precision in dense region (insufficient mantissa)

**Warning signs:**
- One format consistently underperforms another at same bit-width
- High kurtosis tensors perform worse (need more exponent bits)
- Dense, narrow distributions perform worse (need more mantissa bits)

**Detection:**
- Histogram activation/gradient distributions per layer
- Measure effective dynamic range required (99.9th percentile / median)
- Compare performance across format variants (E4M3 vs E5M2, etc.)

**Prevention:**
1. **Distribution analysis**: Profile actual value distributions before choosing format
2. **Adaptive format selection**: Use different formats per layer based on statistics
3. **Comparative testing**: Test multiple formats on representative microbenchmarks

**Your E7M0 risk:**
E7M0 (all exponent, no mantissa) is **extremely mismatched** to neural network distributions. Useful only if you expect:
- Extreme outliers (>7 orders of magnitude range)
- No need for sub-power-of-2 precision
- Highly sparse distributions

This is highly unlikely for transformer training. E7M0 will likely fail due to excessive quantization error in the dense region around zero.

**Your E1M6 format:**
E1M6 (1 exponent, 6 mantissa) provides excellent precision in [0.5, 1.0) range (6 bits = 64 levels) but only 2× dynamic range total. Better suited to **normalized tensors** after LayerNorm or BatchNorm.

**Phase mapping:** Phase 4 (Custom formats) - Analyze distributions BEFORE E7M0 testing to predict failure modes

---

### 8. SwiGLU Outlier Amplification → Progressive Instability

**What goes wrong:**
SwiGLU activation function creates **quadratic growth** in activation magnitudes: y = (W₁x ⊙ σ(W₂x)) ⊙ W₃. The element-wise multiplication amplifies outliers quadratically. FP8 training with SwiGLU causes divergence after 200B+ tokens in large models, with instability worsening over training.

**Why it happens:**
- Weight matrices W₁, W₂ progressively align during training (mechanical artifact)
- Alignment causes larger pre-activation products
- Products exceed FP8 range → overflow → NaN propagation
- Problem emerges **late in training**, not immediately

**Warning signs:**
- SwiGLU output magnitudes increasing over time (track max/std ratio)
- Divergence occurs after X tokens (X varies with model size)
- Disabling SwiGLU quantization prevents divergence
- FFN layers show higher kurtosis than attention layers

**Detection:**
- Track SwiGLU output statistics every N steps
- Monitor weight matrix alignment (cosine similarity between W₁ and W₂)
- Compare FP8 vs BF16 SwiGLU outputs (Δmax should be <1%)

**Prevention:**
1. **Smooth-SwiGLU**: Modified activation that reduces outlier amplification
2. **Selective FP8**: Keep SwiGLU in BF16 (reduces acceleration but prevents divergence)
3. **Dynamic scaling per block**: Finer-grained MXFP8 for FFN layers
4. **Outlier clamping with compensation**: Clamp extreme values, compensate in next layer

**Your scenario:**
If your architecture uses SwiGLU/GeGLU: **Budget for BF16 fallback or Smooth-SwiGLU.** Standard SwiGLU in E7M0 will almost certainly diverge.

**Phase mapping:** Phase 1 (Baseline) - Test SwiGLU stability in BF16 and E4M3 before custom formats

---

## Custom Format Risks (E7M0, E1M6, and Novel Formats)

### 9. E7M0 Powers-of-2: Zero Gradient Regions

**What goes wrong:**
E7M0 has **no mantissa bits**, representing only powers of 2: {..., 0.25, 0.5, 1.0, 2.0, 4.0, ...}. For values in interval (2^k, 2^(k+1)), **all intermediate values collapse to one of the boundaries**. This creates:
- **Zero-gradient regions**: When both input and weight quantize to same power of 2, small perturbations produce no output change (∇ = 0)
- **Gradient staircase**: Discontinuous gradients at power-of-2 boundaries
- **Systematic bias**: Values are not uniformly distributed—bell-shaped distributions become heavily discretized

**Why it happens:**
Neural networks rely on gradients for tiny adjustments. E7M0's exponential spacing makes most updates fall into zero-gradient zones or jump discontinuously.

**Warning signs:**
- Training loss plateaus immediately (within 10 steps)
- Gradient norms near zero despite non-zero loss
- Weight updates become sparse (only some weights update each step)
- Validation accuracy random (no learning occurring)

**Failure probability:** **>90%** for standard transformer training

**Potential mitigations (unlikely to work):**
1. **Extreme learning rate scaling**: LR 10-100× higher to force boundary crossings (risks instability)
2. **Stochastic rounding everywhere**: Introduces gradient signal via variance (but variance may dominate)
3. **Additive powers-of-2 (APoT)**: Represent values as sum of powers-of-2 (e.g., 1.5 = 1.0 + 0.5)
4. **Mixed E7M0/E4M3**: Use E7M0 only for proven robust operations, E4M3 elsewhere

**What would make E7M0 viable:**
- Highly specialized architecture designed for logarithmic quantization
- Sparse activations (most values exactly 0 or powers of 2)
- Multiplicative operations where power-of-2 is efficient (bit shifts)
- NOT standard transformer training

**Phase mapping:** Phase 4 (Custom formats) - Consider E7M0 a **negative result experiment**. Document why it fails for science, not production use.

---

### 10. E1M6: Narrow Range Overflow

**What goes wrong:**
E1M6 (1 exponent bit, 6 mantissa bits) provides excellent precision (64 levels in main range) but **extremely limited dynamic range**:
- Only 2× range: [0.5, 1.0) with 6-bit precision, [1.0, 2.0) with 6-bit precision
- Values <0.5 → underflow to 0
- Values ≥2.0 → overflow to max

For unnormalized tensors (activations before normalization, gradients after backprop), this will cause catastrophic information loss.

**Why it happens:**
1 exponent bit can only represent 2 ranges. Transformers have activation/gradient distributions spanning many orders of magnitude without LayerNorm.

**Warning signs:**
- Sudden NaN/Inf after specific operations
- Gradient clipping at 2.0 or underflow to 0
- Layers before normalization show degraded outputs
- Gradients become all-zero or all-max

**Viable use cases for E1M6:**
- **Post-normalization activations**: LayerNorm outputs have bounded range
- **Attention scores after softmax**: Values in [0, 1]
- **Normalized weights**: Weight tensors after per-channel normalization

**Prevention:**
1. **Per-operation format selection**: E1M6 only after normalization layers
2. **Dynamic range analysis**: Verify 99.9% of values fall in representable range
3. **Hybrid format schedule**: E1M6 for forward, E4M3 for backward
4. **Fallback thresholds**: Auto-switch to wider format if clipping rate >1%

**Phase mapping:** Phase 4 (Custom formats) - E1M6 should be tested on **normalized tensors only**, not general training

---

### 11. Custom Format Implementation Bugs

**What goes wrong:**
When implementing novel FP8 formats, common bugs include:
- **Incorrect rounding modes**: nearest-even vs truncation affects bias
- **Subnormal handling**: Whether gradual underflow is supported
- **Special value encoding**: How NaN, Inf, ±0 are represented
- **Bit layout errors**: Misaligned exponent/mantissa extraction
- **Scaling factor bugs**: Off-by-one in exponent bias calculations

These bugs can cause **silent correctness errors** that appear as poor training performance, not crashes.

**Warning signs:**
- Format behaves differently than mathematical specification predicts
- Quantization errors are biased (average error ≠ 0)
- Round-trip test fails: quantize(dequantize(x)) ≠ quantize(x)
- Bit patterns don't match expected encoding

**Prevention:**
1. **Property-based testing**: Generate random values, verify round-trip properties
2. **Reference implementation**: Compare against IEEE 754 spec or known implementations
3. **Numerical error analysis**: Measure quantization error distribution (should be unbiased)
4. **Boundary value testing**: Test max, min, ±0, subnormals, special values
5. **Bit-level inspection**: Print actual bit patterns for known test values

**Testing protocol for E7M0/E1M6:**
```python
# Test suite must include:
- Round-trip consistency
- Boundary values (max, min representable)
- Powers of 2 (should be exact in E7M0)
- Midpoint rounding (0.75 → 0.5 or 1.0?)
- Negative values (sign bit handling)
- Zero (positive and negative zero?)
- Overflow/underflow behavior
- Stochastic rounding distribution (should match theoretical)
```

**Phase mapping:** Phase 2 (Infrastructure) - Exhaustive testing before any training runs

---

## Manifold-Aware Optimizer Risks

### 12. Stiffness Preconditioning Numerical Instability

**What goes wrong:**
Manifold-aware optimizers with stiffness preconditioning require computing/approximating the Hessian or metric tensor. In low precision:
- **Hessian estimation errors amplify**: Second-order derivatives more sensitive to quantization noise than first-order
- **Condition number explosion**: Ill-conditioned preconditioners in FP8 → NaN in inverse computation
- **Accumulation errors**: Iterative Hessian approximations (KFAC, L-BFGS) accumulate quantization errors over steps

**Why it happens:**
Second-order methods compute curvature information (∂²L/∂θ²), which requires:
1. High precision intermediate values
2. Stable inversion/factorization operations
3. Accurate accumulation over time

FP8's limited precision breaks all three requirements.

**Warning signs:**
- Preconditioner becomes singular (det ≈ 0)
- Inverse Hessian contains NaN/Inf
- Optimizer steps become erratic (loss jumps)
- Curvature estimates diverge from BF16 baseline
- Optimization becomes slower than SGD (bad curvature estimates)

**Prevention:**
1. **FP32 preconditioner state**: Store Hessian approximations in FP32, only quantize gradients
2. **Diagonal approximations**: Use diagonal Hessian (less sensitive than full matrix)
3. **Damping**: Add λI to Hessian before inversion for numerical stability
4. **Selective quantization**: Keep curvature computations in higher precision
5. **Periodic resets**: Re-initialize Hessian estimates every N steps to prevent error accumulation

**Your manifold-aware optimizer:**
If stiffness preconditioning involves:
- Matrix inversions → **Keep in FP32**
- Iterative approximations → **Test error accumulation carefully**
- Riemannian metrics → **Verify metric tensor stays positive definite in FP8**

**Detection strategy:**
- Compare preconditioner eigenvalue spectrum (FP8 vs BF16)
- Track condition number over training
- Verify positive definiteness each step
- Compare effective learning rates (preconditioned gradient norm ratio)

**Phase mapping:** Phase 3 (Format testing) - Validate optimizer with E4M3 BEFORE trying E7M0

---

### 13. Geometric Structure Corruption Under Quantization

**What goes wrong:**
Manifold-aware optimization assumes smooth geometric structure (Riemannian manifolds, orthogonal constraints, etc.). Quantization can **destroy geometric properties**:
- Orthogonality constraints violated (QᵀQ ≠ I after quantization)
- Geodesics become non-smooth (quantization introduces "corners")
- Metric tensor loses positive-definiteness
- Retractions/projections onto manifold fail to converge

**Why it happens:**
Geometric constraints require precise arithmetic. FP8 errors cause:
- Constraint drift: Weights slowly violate manifolds (e.g., drift off Stiefel manifold)
- Non-convergent projections: Iterative projections oscillate instead of converging
- Invalid updates: Quantized exponential map produces points off manifold

**Warning signs:**
- Constraint violation metrics increasing over time
- Projection operations not converging (max iterations reached)
- Orthogonal matrices becoming non-orthogonal (||QᵀQ - I|| >> 0)
- Unexpected gradient directions (not tangent to manifold)

**Detection:**
```python
# For Stiefel manifold (QᵀQ = I):
constraint_violation = torch.norm(Q.T @ Q - torch.eye(d))
# Should be <1e-6 in FP32, but may be >1e-2 in FP8

# For positive definite metric:
eigenvalues = torch.linalg.eigvalsh(metric_tensor)
min_eigenvalue = eigenvalues.min()  # Should be >0
```

**Prevention:**
1. **Periodic re-orthogonalization**: Project back to manifold every N steps
2. **Higher precision for manifold operations**: Retraction/projection in FP32
3. **Constraint monitoring**: Track violation metrics, stop if exceeds threshold
4. **Adaptive precision**: Switch to higher precision when constraint violation detected
5. **Modified retractions**: Use more stable numerical schemes (QR vs SVD)

**Your manifold-aware optimizer:**
- What manifold structure does it assume? (Stiefel, Grassmann, SPD matrices?)
- Which operations MUST preserve structure? (Keep those in FP32)
- Can you verify constraints cheaply? (Add monitoring hooks)

**Phase mapping:** Phase 3 (Format testing) - Establish constraint violation budgets with E4M3 baseline

---

### 14. Curvature-Aware QAT Interactions

**What goes wrong:**
If using curvature-aware gradient estimation (like CAGE framework) with manifold-aware optimization, there's a **compounding complexity**:
- CAGE adds curvature correction to STE gradients
- Manifold optimizer uses curvature for preconditioning
- Two curvature estimates may conflict or amplify errors

The interaction is under-explored in literature.

**Why it happens:**
CAGE gradient: ∇̃ = ∇STE + α∇²L · Δq (curvature correction for quantization step)
Manifold update: Δθ = P⁻¹∇̃ (preconditioned by curvature)

Combined: Δθ = P⁻¹(∇STE + α∇²L · Δq)

If P and ∇²L are estimated differently or from different data, corrections may fight each other.

**Warning signs:**
- Optimizer performs worse than SGD with QAT
- CAGE corrections seem ineffective with manifold optimizer
- Training is unstable when combining both approaches
- Better results with either technique alone than combined

**Prevention:**
1. **Shared curvature estimates**: Use same Hessian approximation for both CAGE and preconditioning
2. **Ablation studies**: Test CAGE-only, manifold-only, and combined
3. **Tuned correction weight**: α parameter may need adjustment with manifold optimizer
4. **Sequential application**: Apply CAGE correction, then manifold update (not simultaneously)

**Your scenario:**
Since you're developing a novel manifold-aware optimizer + testing extreme quantization:
- Start with **manifold optimizer in BF16** (establish baseline)
- Add **quantization with STE** (measure degradation)
- Try **CAGE corrections** if STE insufficient
- Only combine if ablations show benefit

**Phase mapping:** Phase 3 (Format testing) - Manifold optimizer first, QAT techniques second

---

## Hardware and Implementation Pitfalls

### 15. H100 Tensor Core Utilization Failures

**What goes wrong:**
H100 FP8 Tensor Cores provide up to 3,958 TFLOPS, but achieving this requires:
- Matrix dimensions aligned to multiples (typically 8 or 16)
- Correct FP8 format usage (E4M3 for weights/activations, E5M2 for gradients)
- Proper tensor layouts in memory
- Sufficient batch size to saturate SMs

Misalignment causes **fallback to slower CUDA cores**, losing 2-6× speedup with no warning.

**Why it happens:**
Tensor Cores are specialized hardware with strict requirements. Compiler/runtime silently falls back to regular cores if requirements not met.

**Warning signs:**
- FP8 training not faster than BF16 (should be 2-3× faster)
- Low Tensor Core utilization in profiler (<50%)
- Memory bandwidth bottleneck (not compute bottleneck)
- Performance varies with batch size non-linearly

**Detection:**
```bash
# Profile with NVIDIA tools:
nsys profile --trace=cuda,nvtx python train.py
# Check Tensor Core utilization in report

# Or use PyTorch profiler:
torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
)
```

**Prevention:**
1. **Dimension alignment**: Ensure d_model, d_ff divisible by 16
2. **Batch size tuning**: Test batch sizes [8, 16, 32, 64] for optimal throughput
3. **Transformer Engine**: Use NVIDIA's Transformer Engine for automatic management
4. **Memory layout**: Ensure tensors are contiguous and properly strided

**Your ≤50M param constraint:**
With small models, **memory bandwidth may dominate over compute**. FP8 speedup requires:
- Sufficient arithmetic intensity (FLOP/byte ratio)
- Large enough matrices to amortize Tensor Core launch overhead

You may see <2× speedup or none at all if model is too small. This is **expected**, not a bug.

**Phase mapping:** Phase 2 (Infrastructure) - Benchmark Tensor Core utilization with standard formats first

---

### 16. Batch Size and Quantization Interactions

**What goes wrong:**
Quantization's performance benefit scales with batch size (larger batches → better Tensor Core utilization), but training stability degrades with small batches (stochastic rounding variance). Your ≤$20 budget constrains batch size.

**The tradeoff:**
- Large batch: Better FP8 performance, lower SR variance, but higher memory cost
- Small batch: Fits in memory, but FP8 may not accelerate, high gradient variance

**Warning signs:**
- FP8 slower than BF16 at your target batch size
- Training unstable at batch size needed for FP8 speedup
- Gradient variance increasing as you increase batch size for performance

**Prevention:**
1. **Gradient accumulation**: Simulate larger batch for stability while keeping memory low
2. **Per-sample gradient clipping**: Reduce variance at small batch sizes
3. **Profiling-guided batch size**: Find sweet spot between memory, speed, stability
4. **Adaptive batch size**: Start large for stability, decrease as training stabilizes

**Your scenario:**
With single H100 and ≤50M params:
- Start with largest batch that fits in memory
- Profile FP8 vs BF16 speedup at that batch size
- If no speedup, consider if FP8 is worth the stability risk (probably not for baseline)

**Phase mapping:** Phase 2 (Infrastructure) - Establish batch size baseline before format experiments

---

## Detection and Monitoring Strategy

### 17. Insufficient Monitoring → Undetected Failure Modes

**What goes wrong:**
FP8 failures are often **subtle and delayed**:
- Kurtosis increases precede divergence by thousands of steps
- Gradient bias accumulates slowly
- Constraint violations drift over time
- Critical data size effects appear after specific token count

Without comprehensive monitoring, you'll waste GPU time on doomed runs.

**Essential metrics to track:**

**Per-layer, every 100 steps:**
- Activation statistics: min, max, mean, std, kurtosis
- Gradient statistics: norm, max, clipping rate
- Quantization error: ||x_fp8 - x_bf16|| / ||x_bf16||
- Attention entropy: per-head average
- Weight statistics: norm, max eigenvalue

**Global, every step:**
- Training loss, validation loss
- Gradient norm (global)
- Learning rate
- NaN/Inf detection flags

**Optimizer-specific (if manifold-aware):**
- Constraint violation: ||g(θ)|| where g is constraint function
- Preconditioner condition number
- Effective learning rate: ||P⁻¹∇|| / ||∇||

**Format-specific (E7M0/E1M6):**
- Quantization histogram: distribution of quantized values
- Zero-gradient region percentage: fraction of params with ∇=0
- Dynamic range utilization: (max - min) / representable_range

**Implementation:**
```python
class FP8Monitor:
    def __init__(self, log_every=100):
        self.metrics = defaultdict(list)
        self.log_every = log_every

    def log_activation(self, name, tensor, step):
        if step % self.log_every != 0:
            return
        self.metrics[f"{name}/max"].append(tensor.abs().max().item())
        self.metrics[f"{name}/kurtosis"].append(kurtosis(tensor))
        # ... etc

    def check_divergence(self, step):
        # Early warning system
        if self.metrics["kurtosis"][-1] > 10:
            warnings.warn(f"Step {step}: High kurtosis detected")
        # ... other checks
```

**Phase mapping:** Phase 1 (Baseline) - Build monitoring infrastructure before experiments

---

## Prevention Strategies Summary

### Phase 1: Baseline Validation (Before Custom Formats)

**Goal:** Establish stable BF16 baseline and validate standard FP8 (E4M3/E5M2)

**Must complete:**
1. Implement comprehensive monitoring system
2. Verify E4M3/E5M2 implementations (maxval bug check)
3. Establish BF16 baseline metrics
4. Test attention entropy stability
5. Validate SwiGLU behavior (if used)
6. Profile H100 Tensor Core utilization

**Success criteria:**
- BF16 training converges to expected loss
- E4M3/E5M2 training within 1% of BF16 loss
- No NaN/Inf during 1000+ step run
- Monitoring captures all essential metrics
- Clear understanding of which operations are numerically sensitive

**Estimated duration:** 2-3 days of experimentation

---

### Phase 2: Infrastructure for Custom Formats

**Goal:** Build robust quantization infrastructure and testing suite

**Must complete:**
1. Implement E7M0 and E1M6 quantization functions
2. Property-based testing suite (round-trip, boundaries, special values)
3. Per-tensor scaling system with amax tracking
4. Stochastic rounding implementation with configurable modes
5. Format selection framework (per-layer format assignment)

**Success criteria:**
- All quantization tests pass
- Round-trip errors within theoretical bounds
- Scaling system prevents overflow/underflow in synthetic tests
- Can switch formats per-layer without crashes

**Estimated duration:** 3-5 days of implementation + testing

---

### Phase 3: Format Testing with Standard Optimizer

**Goal:** Isolate format effects from optimizer effects

**Test sequence:**
1. **E4M3 baseline** (reproduce literature results)
2. **E5M2 for gradients** (standard mixed-precision)
3. **E1M6 on normalized tensors only** (post-LayerNorm activations)
4. **E7M0 negative result experiment** (expect failure, document why)

**For each format:**
- Run 3 seeds for statistical confidence
- Monitor all metrics from Phase 1
- Compare loss curves to BF16 baseline
- Measure final perplexity/accuracy
- Profile performance (speedup vs BF16)

**Success criteria:**
- E4M3/E5M2 within 2% of BF16 (matches literature)
- E1M6 viable for specific layers (or documented failure)
- E7M0 failure modes documented (gradient sparsity, zero-regions, etc.)
- Clear evidence for which formats are viable

**Estimated duration:** 5-7 days of experiments

---

### Phase 4: Manifold-Aware Optimizer Integration

**Goal:** Combine custom formats with novel optimizer safely

**Incremental approach:**
1. **Manifold optimizer in BF16** (establish baseline vs SGD/Adam)
2. **Manifold + E4M3** (standard quantization)
3. **Manifold + E1M6** (if E1M6 viable from Phase 3)
4. **Manifold + CAGE corrections** (if needed)

**Critical checks:**
- Manifold constraint violations (orthogonality, positive-definiteness)
- Preconditioner stability (condition number, eigenvalues)
- Curvature estimation errors (FP8 vs BF16 Hessian approximations)
- Optimization efficiency (steps to convergence vs SGD)

**Success criteria:**
- Manifold optimizer converges in BF16
- Quantized version maintains constraint violations <threshold
- No optimizer-specific divergences
- Documented interaction effects (positive or negative)

**Estimated duration:** 4-6 days of experiments

---

## Phase Mapping: What Can Go Wrong When

| Phase | Primary Risks | Detection | Mitigation |
|-------|---------------|-----------|------------|
| **Phase 1: Baseline** | E5M2 maxval bug, monitoring gaps, attention entropy collapse | NaN injection, entropy <1.0, loss divergence | Unit tests, BF16 attention, σReparam |
| **Phase 2: Infrastructure** | Implementation bugs in E7M0/E1M6, scaling errors | Round-trip test failures, biased quantization error | Property-based testing, reference implementations |
| **Phase 3: Format Testing** | E7M0 immediate collapse, E1M6 overflow, SR variance explosion | Zero gradients, loss plateau, high variance | Expect E7M0 failure, restrict E1M6 to normalized tensors, split SR strategy |
| **Phase 4: Optimizer Integration** | Geometric structure corruption, curvature estimation errors, Hessian instability | Constraint violations, preconditioner singular, erratic updates | FP32 preconditioner, damping, periodic re-orthogonalization |
| **Phase 5: Scale Testing** | Critical data size degradation, progressive SwiGLU instability | Validation loss increases, kurtosis growth | Early stopping, mixed-precision checkpointing |

---

## Open Questions Requiring Experimentation

These cannot be answered from literature alone—your experiments will provide novel insights:

1. **E7M0 gradient sparsity:** What fraction of gradients are exactly zero due to powers-of-2 quantization? Does this make training impossible or just inefficient?

2. **E1M6 sweet spot:** Which specific layers/operations can use E1M6 safely? Hypothesis: post-LayerNorm activations, post-Softmax attention weights.

3. **Manifold + extreme quantization:** Does stiffness preconditioning provide ANY benefit under E7M0, or does quantization destroy curvature information?

4. **Stochastic rounding modes:** For powers-of-2 formats, does stochastic rounding help or hurt? Variance may dominate.

5. **Hybrid format schedules:** Can you start training in BF16, switch to E4M3 mid-training, then E1M6 for final finetuning?

6. **Quantization error vs optimization error:** For E7M0, is poor performance due to quantization error overwhelming gradient signal, or optimization landscape becoming pathological?

---

## Sources

### FP8 Training Fundamentals
- [FP8 Training Notes (Harold Benoit)](https://haroldbenoit.com/notes/ml/engineering/precision/fp8/fp8-training)
- [FP8 Mixed-Precision Training (EmergentMind)](https://www.emergentmind.com/topics/fp8-mixed-precision-training-framework)
- [NVIDIA FP8 Technical Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [Transformer Engine FP8 Primer (NVIDIA Docs)](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

### Critical Research Papers (2025-2026)
- [To FP8 and Back Again: Quantifying Reduced Precision Effects (OpenReview)](https://openreview.net/forum?id=pNgyXuGcx4)
- [FP8-LM: Training FP8 Large Language Models (arXiv)](https://arxiv.org/pdf/2310.18313)
- [Scaling FP8 Training to Trillion-Token LLMs (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/f48b5133e89854a9e97cc22a6db83f25-Paper-Conference.pdf)
- [DeepSeek FP8 Training Analysis (Medium)](https://dataturbo.medium.com/deepseek-technical-analysis-5-fp8-training-ff34768727b8)

### Instability and Failure Modes
- [Characterization and Mitigation of Training Instabilities in Microscaling Formats (arXiv)](https://arxiv.org/html/2506.20752v1)
- [Low-Precision Training of LLMs: Methods, Challenges, Opportunities (arXiv)](https://arxiv.org/pdf/2505.01043)
- [Stabilizing Transformer Training by Preventing Attention Entropy Collapse (Apple ML Research)](https://machinelearning.apple.com/research/stabilizing-transformer-training)
- [What Happens During the Loss Plateau? Understanding Abrupt Learning (OpenReview)](https://openreview.net/forum?id=tnTM6JJuLi)

### Quantization Techniques
- [CAGE: Curvature-Aware Gradient Estimation for QAT (arXiv)](https://arxiv.org/html/2510.18784v1)
- [Direct Quantized Training with Stochastic Rounding (arXiv)](https://arxiv.org/html/2412.04787v1)
- [FP4 All the Way: Fully Quantized Training (arXiv)](https://arxiv.org/html/2505.19115)
- [Training with Fewer Bits: Unlocking Edge LLMs with Stochastic Rounding (arXiv)](https://arxiv.org/html/2511.00874)
- [Additive Powers-of-Two Quantization (ICLR 2020)](https://openreview.net/forum?id=BkgXT24tDS)

### Scaling and Loss Scaling
- [Per-Tensor and Per-Block Scaling Strategies (NVIDIA Blog)](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Unified FP8: Moving Beyond Mixed Precision for MoE RL (LMSYS)](https://lmsys.org/blog/2025-11-25-fp8-rl/)

### Second-Order and Geometric Optimization
- [Second-Order Fine-Tuning Without Pain (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/6bf82cc56a5fa0287c438baa8be65a70-Paper-Conference.pdf)
- [Riemannian Optimization for LoRA on Stiefel Manifold (EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.1143.pdf)
- [Survey of Geometric Optimization for Deep Learning (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3708498)

### Hardware and Performance
- [H100 FP8 Guide (Uvation)](https://uvation.com/articles/guide-to-h100-fp8)
- [Serving Quantized LLMs on H100 Tensor Cores (Databricks)](https://www.databricks.com/blog/serving-quantized-llms-nvidia-h100-tensor-core-gpus)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

**Confidence Assessment:** HIGH for standard FP8 pitfalls, MEDIUM for custom format predictions (based on theory + analogous research), LOW for manifold-optimizer interaction (novel combination, limited literature)
