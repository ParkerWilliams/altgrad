# AltGrad Test Matrix

This document defines all format x optimizer x layer combinations being tested in the AltGrad FP8 training research.

## FP8 Formats

| Format | Exponent | Mantissa | Range | Precision | Use Case |
|--------|----------|----------|-------|-----------|----------|
| E0M7 | 0 | 7 | [-127/128, 127/128] | Fixed-point | Post-LayerNorm normalized tensors |
| E1M6 | 1 | 6 | Two scales | High precision | Attention weights |
| E3M4 | 3 | 4 | ~0.06 - 124 | Balanced | General MLP layers |
| E5M2 | 5 | 2 | ~1e-7 - 57344 | Wide range | Standard FP8 baseline |
| E7M0 | 7 | 0 | Powers of 2 only | Minimal | Extreme experiment (expected failure) |

### Format Transfer Functions

- **E0M7**: Fixed-point with uniform grid spacing (stiffness = 1/128 everywhere)
- **E1M6**: Two-scale format with bias=0
- **E3M4**: bias=1 for range ~0.06 to ~124 (verified in Phase 1)
- **E5M2**: Standard FP8 with bias=15, max value 57344
- **E7M0**: Pure power-of-2 values, gradient update almost always stalls

## Optimizers

| Optimizer | Master Weights | Update Mechanism | Key Parameters | Flip Tracking |
|-----------|---------------|------------------|----------------|---------------|
| AdamW | Implicit FP32 | Standard Adam | lr, betas, weight_decay | No |
| ManifoldAdamW | Implicit FP32 | Stiffness-preconditioned | mantissa_bits, max_stiffness | Yes (bit_position) |
| GridOptim | Explicit FP32 | Rung-based stochastic | scale, momentum, rung_clip | Yes (returns flips, updates) |

### Optimizer Details

**ManifoldAdamW** (Phase 5):
- Multiplies gradients by local stiffness S = 2^(floor(log2|w|) - M)
- Moves weights by consistent ULP counts rather than fixed real values
- Tracks bit_position for latent integer state monitoring

**GridOptim** (Phase 8):
- Builds explicit grid from FP8 representable values: `torch.arange(-128,128).view(DTYPE)`
- Stochastic rounding: `floor(v_rungs + rand_like(v_rungs))`
- Rung clipping: `clamp(v_rungs, -10, 10)` prevents NaN at grid boundaries
- Returns both flips (FP8 changed) and updates (non-zero gradient)

## Layer Types and Monitoring

| Layer Type | Pattern | Critical | Rank Threshold | Notes |
|------------|---------|----------|----------------|-------|
| Classifier | `lm_head` | Yes | 0.15 (stricter) | Output-critical, early warning |
| Attention output | `c_proj` | Yes | 0.15 (stricter) | High-impact on representations |
| Attention weights | `attn.qkv` | No | 0.30 (standard) | Internal projections |
| MLP layers | `mlp.fc*` | No | 0.30 (standard) | Feed-forward layers |
| Embeddings | `wte`, `wpe` | No | 0.30 (standard) | Input projections |

### Monitoring Metrics

**Flip Metrics** (Phase 7 + Phase 8):
- `flip_count`: Number of FP8 values that changed
- `update_count`: Number of non-zero gradients (attempted updates)
- `stall_ratio`: 1 - (flips / updates) - measures gradient effectiveness
  - 0.0 = all updates cause flips (ideal)
  - 1.0 = no updates cause flips (complete stall)

**Rank Metrics** (Phase 7 + Phase 8):
- `stable_rank`: ||W||_F^2 / ||W||_2^2 - effective dimension
- `effective_rank`: exp(entropy of normalized singular values)
- `spectral_norm`: Largest singular value ||W||_2

## Test Combinations

### Primary Experiments

| Format | Optimizer | Layer Config | Status | Priority |
|--------|-----------|--------------|--------|----------|
| E5M2 | AdamW | Uniform | Baseline | P0 |
| E5M2 | ManifoldAdamW | Uniform | Comparison | P0 |
| E5M2 | GridOptim | Uniform | Reference | P0 |
| E3M4 | GridOptim | Uniform | Higher precision | P1 |
| E3M4 | ManifoldAdamW | Uniform | Comparison | P1 |
| E1M6 | GridOptim | Post-LN only | Narrow range | P2 |
| E0M7 | GridOptim | Post-LN only | Fixed-point | P2 |

### Extreme Experiments (Expected Failure)

| Format | Optimizer | Expected Outcome | Purpose |
|--------|-----------|------------------|---------|
| E7M0 | GridOptim | Training collapse | Quantify power-of-2 limitation |
| E7M0 | ManifoldAdamW | Training collapse | Document stiffness behavior |

### Mixed Precision Configurations

| Layer | Format | Rationale |
|-------|--------|-----------|
| lm_head | BF16 | Preserve classifier precision |
| c_proj | BF16 | Preserve attention output |
| attn.qkv | E5M2 | Standard quantization |
| mlp.fc1, fc2 | E3M4 | Higher internal precision |

## Metrics Collection

All experiments log the following to W&B:

**Per-step metrics:**
- loss, gradient_norm
- flip_count (per layer, total)
- update_count (per layer, total)
- stall_ratio (per layer)
- bit_position_mean/std (ManifoldAdamW only)

**Per-log-interval metrics (every 100 steps):**
- stable_rank (per layer)
- effective_rank (per layer)
- spectral_norm (per layer)
- rank_warnings (list)

**Per-epoch metrics:**
- epoch_loss, epoch_flips_total
- epoch_stall_ratio_mean

## Success Criteria

1. **E5M2 baseline**: Training converges, loss decreases monotonically
2. **Manifold benefit**: ManifoldAdamW shows different dynamics than AdamW
3. **Grid reference**: GridOptim produces stable training with explicit flip control
4. **E7M0 failure**: Documented collapse point and gradient sparsity
5. **Classifier health**: lm_head rank warnings fire before other layers

---
*Generated: Phase 8 - Update Metrics & Test Matrix*
*Last updated: 2026-01-26*
