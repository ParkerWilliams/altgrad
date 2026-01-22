# Phase 4: Custom Format Testing - Research

**Researched:** 2026-01-21
**Domain:** FP8 format experimentation, stability interventions, quantization diagnostics
**Confidence:** MEDIUM to HIGH

## Summary

Phase 4 implements systematic testing of exotic FP8 formats (E3M4, E1M6, E7M0, E0M7) against the E5M2 baseline, with stability interventions to handle format-specific failure modes. The phase requires implementing partition-relative gradient clipping (STAB-05), emergency mantissa shift (STAB-06), and four diagnostic requirements (DIAG-01 through DIAG-04).

The codebase already has strong foundations: `BitStallDetector` for DIAG-04 (ULP-related), `FP8Format.max_representable_value` for dynamic range calculations, and `WandbTracker` for alert infrastructure. The key additions are:
1. **Partition-relative clipping**: Adapt clip thresholds based on format's dynamic range (E5M2: 57344, E3M4: ~124, E7M0: ~2^64 but with severe precision loss)
2. **Emergency mantissa shift**: Runtime format switching when NaN or bit-stall thresholds exceeded
3. **Stiffness field visualization**: Track per-weight quantization "stiffness" S = 2^(floor(log2|w|) - M) over training
4. **Grid alignment measurement**: Distance from each weight to its nearest quantization level

**Primary recommendation:** Build on existing `BitStallDetector` and `compute_scale` patterns. Implement interventions as training callbacks, not trainer modifications. Store diagnostics in a new `altgrad/quantization/advanced_diagnostics.py` module.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | Core tensor operations, `torch.nextafter` for ULP | Already in use |
| matplotlib | 3.x | Stiffness field heatmaps, diagnostic plots | Standard visualization |
| seaborn | 0.13+ | Enhanced heatmaps with annotations | Cleaner heatmap API |
| numpy | 1.26+ | Numerical computations for analysis | Already in use |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| wandb | 0.16+ | Metric logging, image logging for heatmaps | Already integrated |
| PyYAML | 6.0+ | Experiment configs | Already in use |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib heatmaps | plotly interactive | plotly adds complexity, matplotlib sufficient for static reports |
| Custom ULP computation | `torch.nextafter` | Use nextafter - it's the correct IEEE 754 approach |

**Installation:**
```bash
# matplotlib and seaborn likely already available; if not:
pip install matplotlib seaborn
```

## Architecture Patterns

### Recommended Project Structure
```
altgrad/
  quantization/
    advanced_diagnostics.py    # NEW: DIAG-01 to DIAG-04
    stability.py               # NEW: STAB-05, STAB-06
  training/
    format_runner.py           # NEW: Experiment orchestration
experiments/
  configs/
    e3m4_uniform.yaml          # Format-specific configs
    e1m6_uniform.yaml
    e7m0_uniform.yaml
    e0m7_uniform.yaml
  reports/
    failure_analysis.md        # NEW: Generated failure reports
```

### Pattern 1: Format-Aware Gradient Clipping
**What:** Adjust gradient clip threshold based on format's dynamic range
**When to use:** When overflow rate exceeds threshold (1%)
**Example:**
```python
# Source: Derived from NVIDIA Transformer Engine patterns + AGGC research
class PartitionRelativeClipper:
    """Clip gradients relative to format's dynamic range."""

    def __init__(self, format: FP8Format, base_clip: float = 1.0):
        self.format = format
        # Scale clip threshold by format's dynamic range ratio vs E5M2
        e5m2_max = 57344.0  # E5M2 max representable
        format_max = format.max_representable_value
        # Formats with smaller range need proportionally smaller clips
        self.clip_threshold = base_clip * (format_max / e5m2_max)

    def clip_if_needed(self, model: nn.Module, overflow_rate: float) -> bool:
        """Apply clipping only when overflow detected."""
        if overflow_rate > 0.01:  # 1% threshold from CONTEXT.md
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.clip_threshold
            )
            return True
        return False
```

### Pattern 2: Emergency Format Shift via Callback
**What:** Monitor NaN/stall rates and trigger format change
**When to use:** Persistent NaN (3+ batches) or stall rate > 50%
**Example:**
```python
# Source: Inspired by NVIDIA's selective BF16 fallback strategy
class EmergencyMantissaShift:
    """Shift to higher-mantissa format on instability."""

    FORMAT_FALLBACK = {
        "E7M0": "E5M2",  # Powers-of-2 -> standard FP8
        "E3M4": "E5M2",  # Moderate range -> wide range
        "E1M6": "E3M4",  # Narrow range -> moderate
        "E0M7": "E3M4",  # Fixed-point -> floating
    }

    def __init__(self, nan_patience: int = 3, stall_threshold: float = 0.5):
        self.consecutive_nans = 0
        self.nan_patience = nan_patience
        self.stall_threshold = stall_threshold

    def check_and_shift(
        self,
        current_format: str,
        has_nan: bool,
        stall_rate: float
    ) -> Optional[str]:
        """Return new format name if shift needed, else None."""
        if has_nan:
            self.consecutive_nans += 1
        else:
            self.consecutive_nans = 0

        should_shift = (
            self.consecutive_nans >= self.nan_patience or
            stall_rate > self.stall_threshold
        )

        if should_shift and current_format in self.FORMAT_FALLBACK:
            self.consecutive_nans = 0  # Reset counter
            return self.FORMAT_FALLBACK[current_format]
        return None
```

### Pattern 3: ULP Statistics via torch.nextafter
**What:** Measure how many ULPs (units in last place) each weight update moves
**When to use:** DIAG-04 requirement
**Example:**
```python
# Source: IEEE 754 standard, Python math.ulp() equivalent
def compute_ulp_distance(before: Tensor, after: Tensor) -> Tensor:
    """Compute how many ULPs each weight moved."""
    # Get the ULP at each position (distance to next representable value)
    ulp = torch.abs(torch.nextafter(before, torch.full_like(before, float('inf'))) - before)
    # Compute distance in ULP units
    distance = torch.abs(after - before) / ulp.clamp(min=1e-45)  # Avoid div by 0
    return distance

def ulp_statistics(before: Tensor, after: Tensor) -> Dict[str, float]:
    """Return ULP movement statistics."""
    ulp_dist = compute_ulp_distance(before, after)
    return {
        "ulp_mean": ulp_dist.mean().item(),
        "ulp_median": ulp_dist.median().item(),
        "ulp_max": ulp_dist.max().item(),
        "ulp_zero_frac": (ulp_dist == 0).float().mean().item(),  # Stalled updates
    }
```

### Pattern 4: Stiffness Field Computation
**What:** Per-weight stiffness S = 2^(floor(log2|w|) - M) tracking over training
**When to use:** DIAG-01 requirement
**Example:**
```python
# Source: Project REQUIREMENTS.md MANI-01 definition
def compute_stiffness_field(weights: Tensor, mantissa_bits: int) -> Tensor:
    """Compute stiffness factor for each weight.

    Stiffness S = 2^(floor(log2|w|) - M) represents the minimum
    meaningful update size at each weight's magnitude.
    """
    # Handle zero weights (infinite stiffness conceptually, use large value)
    abs_w = weights.abs().clamp(min=1e-45)
    log2_w = torch.floor(torch.log2(abs_w))
    stiffness = torch.pow(2.0, log2_w - mantissa_bits)
    # Zero weights have undefined stiffness; mark as NaN or max
    stiffness = torch.where(weights == 0, torch.tensor(float('nan')), stiffness)
    return stiffness
```

### Pattern 5: Quantization Grid Alignment
**What:** Measure distance from each weight to nearest FP8 value
**When to use:** DIAG-02 requirement
**Example:**
```python
# Source: Derived from quantize operation
def grid_alignment_error(weights: Tensor, format: FP8Format, scale: Tensor) -> Tensor:
    """Compute distance to nearest quantization grid point."""
    from altgrad.quantization import quantize
    quantized = quantize(weights, format, scale)
    # Absolute error is distance to nearest grid point
    return torch.abs(weights - quantized)

def grid_alignment_statistics(weights: Tensor, format: FP8Format, scale: Tensor) -> Dict[str, float]:
    """Return grid alignment statistics."""
    error = grid_alignment_error(weights, format, scale)
    return {
        "grid_error_mean": error.mean().item(),
        "grid_error_max": error.max().item(),
        "grid_error_std": error.std().item(),
        "on_grid_frac": (error < 1e-10).float().mean().item(),  # Exactly on grid
    }
```

### Anti-Patterns to Avoid
- **Always-on clipping:** Only activate partition-relative clipping when overflow detected, not by default
- **Hardcoded format fallbacks:** Make fallback chain configurable per experiment
- **Blocking visualization:** Generate heatmaps asynchronously or at end of training, not every step

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ULP computation | Custom bit manipulation | `torch.nextafter(x, inf) - x` | IEEE 754 compliant, handles denormals |
| Heatmap visualization | Manual matplotlib grid | `seaborn.heatmap()` with annotations | Better defaults, auto colorbar |
| NaN detection | Manual float checks | `torch.isnan()`, `math.isnan()` | Faster, handles edge cases |
| Gradient clipping | Custom norm computation | `torch.nn.utils.clip_grad_norm_` | Battle-tested, efficient |
| Config serialization | Custom YAML handling | Existing `TrainConfig` pattern | Already works, has YAML support |

**Key insight:** The existing codebase patterns (diagnostics, metrics, callbacks) should be extended, not replaced. New functionality fits into established module boundaries.

## Common Pitfalls

### Pitfall 1: E7M0 Stall Rate Misinterpretation
**What goes wrong:** E7M0 (powers of 2 only) will show near-100% bit-stall for normal gradient magnitudes, leading to premature format abandonment
**Why it happens:** E7M0 has zero mantissa bits, so any update smaller than a power of 2 rounds to zero
**How to avoid:** Expected behavior for E7M0 - document as scientific negative result, don't "fix" it
**Warning signs:** Test already shows >70% stall rate for E7M0 with moderate gradients (test_diagnostics.py line 51-55)

### Pitfall 2: E1M6 Overflow in Attention
**What goes wrong:** E1M6 has very narrow range (approximately [-2, 2]), attention scores often exceed this
**Why it happens:** 1 exponent bit means only 2 scales available
**How to avoid:** Selective application - use E1M6 only on post-LayerNorm tensors which are normalized
**Warning signs:** Immediate NaN on first forward pass

### Pitfall 3: Scale Mismatch During Format Shift
**What goes wrong:** Emergency mantissa shift changes format but keeps old scale factor
**Why it happens:** Scale is computed from `format.max_representable_value`, which differs between formats
**How to avoid:** Reset `AmaxHistory` and recompute scale after format shift
**Warning signs:** Overflow spike immediately after format shift

### Pitfall 4: Stiffness Computation for Zero Weights
**What goes wrong:** `log2(0)` is undefined, causes NaN in stiffness field
**Why it happens:** Zero weights exist in initialized models
**How to avoid:** Clamp to minimum value or mark zero-weight stiffness as NaN explicitly
**Warning signs:** NaN values appearing in stiffness heatmaps

### Pitfall 5: Checkpoint Format Incompatibility
**What goes wrong:** Checkpoint saved with E7M0, trying to load into E5M2 model
**Why it happens:** Emergency shift happened, checkpoint was saved before shift
**How to avoid:** Save current format name in checkpoint metadata, validate on load
**Warning signs:** Shape or value mismatches on checkpoint restore

## Code Examples

Verified patterns from official sources and existing codebase:

### Gradient-Stiffness Correlation (DIAG-03)
```python
# Source: Derived from project requirements
def gradient_stiffness_correlation(
    weights: Tensor,
    gradients: Tensor,
    mantissa_bits: int
) -> Dict[str, float]:
    """Analyze correlation between gradient magnitude and stiffness.

    High correlation indicates gradients align with quantization grid.
    """
    stiffness = compute_stiffness_field(weights, mantissa_bits)
    grad_mag = gradients.abs()

    # Flatten and remove NaN stiffness (zero weights)
    valid_mask = ~torch.isnan(stiffness)
    s_flat = stiffness[valid_mask].flatten()
    g_flat = grad_mag[valid_mask].flatten()

    # Compute Pearson correlation
    s_centered = s_flat - s_flat.mean()
    g_centered = g_flat - g_flat.mean()
    correlation = (s_centered * g_centered).sum() / (
        s_centered.norm() * g_centered.norm() + 1e-10
    )

    # Ratio of gradient to stiffness (< 1 means likely stall)
    ratio = g_flat / s_flat.clamp(min=1e-10)

    return {
        "grad_stiff_correlation": correlation.item(),
        "grad_stiff_ratio_mean": ratio.mean().item(),
        "grad_below_stiffness_frac": (ratio < 1.0).float().mean().item(),
    }
```

### Stiffness Field Visualization (DIAG-01)
```python
# Source: matplotlib/seaborn heatmap patterns
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_stiffness_field(
    stiffness: Tensor,
    layer_name: str,
    step: int,
    output_path: str
) -> None:
    """Create heatmap of stiffness values for a layer."""
    # Reshape to 2D if needed (for visualization)
    s_np = stiffness.detach().cpu().numpy()
    if s_np.ndim == 1:
        # Reshape 1D to approximately square
        side = int(np.ceil(np.sqrt(len(s_np))))
        pad_len = side * side - len(s_np)
        s_np = np.pad(s_np, (0, pad_len), constant_values=np.nan)
        s_np = s_np.reshape(side, side)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        np.log2(s_np),  # Log scale for visualization
        cmap='viridis',
        cbar_kws={'label': 'log2(Stiffness)'},
        ax=ax
    )
    ax.set_title(f'{layer_name} Stiffness Field (Step {step})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### Format Experiment Runner
```python
# Source: Existing run_experiment.py pattern
def run_format_experiment(
    format_name: str,
    config: TrainConfig,
    data_dir: str,
    device: str,
    max_steps: int = 500
) -> Dict[str, Any]:
    """Run single format experiment with failure capture."""
    # Modify config for this format
    config.fp8_format = format_name
    config.use_fp8 = True

    # Initialize stability interventions
    clipper = PartitionRelativeClipper(FORMAT_REGISTRY[format_name])
    shifter = EmergencyMantissaShift(nan_patience=3, stall_threshold=0.5)

    # Create model and trainer
    model = GPT(GPTConfig(...)).to(device)
    trainer = Trainer(config, model, data_dir, device)

    results = {
        "format": format_name,
        "steps_completed": 0,
        "final_loss": None,
        "failure_step": None,
        "failure_reason": None,
        "metrics_history": [],
    }

    try:
        for step in range(max_steps):
            x, y = get_batch(...)
            metrics = trainer.train_step(x, y)
            results["metrics_history"].append(metrics)

            # Check for failure
            if math.isnan(metrics["loss"]):
                results["failure_step"] = step
                results["failure_reason"] = "NaN loss"
                # Save checkpoint for forensics
                trainer.checkpoint_manager.save_on_anomaly(...)
                break

            results["steps_completed"] = step + 1
            results["final_loss"] = metrics["loss"]

    except Exception as e:
        results["failure_reason"] = str(e)

    return results
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Global gradient clipping | Per-group adaptive clipping (AGGC) | 2025 | Eliminates "spill-over" effect between stable/unstable groups |
| Single FP8 format for all | Hybrid E4M3/E5M2 (forward/backward) | 2023-2024 | Standard practice for FP8 training now |
| Per-tensor scaling only | MXFP8 block scaling (Blackwell) | 2025 | 32-element blocks enable E4M3 everywhere (not H100 applicable) |
| Fixed clip thresholds | EMA-based adaptive bounds | 2025 (AGGC) | More robust to architecture heterogeneity |

**Deprecated/outdated:**
- Static loss scaling alone: Now combined with delayed per-tensor scaling
- Universal format across layers: Per-layer format selection is standard research direction

## Open Questions

Things that couldn't be fully resolved:

1. **Mantissa Shift Direction**
   - What we know: Shifting E->M increases precision, M->E increases range
   - What's unclear: For this project, should "mantissa shift" mean E7M0->E5M2 or E5M2->E3M4?
   - Recommendation: Interpret as "increase mantissa" fallback chain (E7M0->E5M2->E3M4) based on CONTEXT.md intent

2. **Stiffness Field Sampling Frequency**
   - What we know: Computing per weight adds overhead
   - What's unclear: How often to sample without slowing training significantly
   - Recommendation: Sample every 50 steps, or at checkpoints only

3. **E0M7 Fixed-Point Handling**
   - What we know: E0M7 is pure fixed-point [-1, 1), not floating-point
   - What's unclear: Whether stiffness formula S = 2^(log2|w| - M) applies (no exponent)
   - Recommendation: For E0M7, stiffness is constant (1/128) since uniform grid spacing

4. **Mixed Format Fallback Strategy**
   - What we know: User decision says "retry with selective application"
   - What's unclear: Exact layer selection heuristic
   - Recommendation: Keep attention layers in failed format's fallback, shift MLP first

## Sources

### Primary (HIGH confidence)
- [torch.nextafter documentation](https://docs.pytorch.org/docs/stable/generated/torch.nextafter.html) - ULP computation
- [NVIDIA Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) - E4M3/E5M2 max values, scaling strategies
- [NVIDIA FP8 Introduction Blog](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/) - Dynamic ranges, stability techniques
- Existing codebase: `altgrad/quantization/formats.py`, `altgrad/quantization/diagnostics.py`, `altgrad/training/metrics.py`

### Secondary (MEDIUM confidence)
- [AGGC: Adaptive Group Gradient Clipping](https://arxiv.org/html/2601.11864) - Per-group clipping with EMA-based bounds
- [To FP8 and Back Again](https://arxiv.org/abs/2405.18710) - FP8 training stability analysis
- [FP8 Quantization: The Power of the Exponent](https://arxiv.org/abs/2208.09225) - Format tradeoffs

### Tertiary (LOW confidence)
- [Visualizing Loss Landscape](https://arxiv.org/abs/1712.09913) - General visualization approach (not specific to stiffness)
- WebSearch results on adaptive quantization - Community patterns, not verified with official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses existing PyTorch/matplotlib patterns
- Architecture: MEDIUM - Novel diagnostic requirements, patterns derived from codebase
- Stability interventions: MEDIUM - Based on research papers, implementation details custom
- Pitfalls: MEDIUM to HIGH - Some verified by existing tests, others from research

**Research date:** 2026-01-21
**Valid until:** 30 days (stable domain, established patterns)
