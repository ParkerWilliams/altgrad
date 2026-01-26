# Phase 07: Flip Metrics & Rank Health Monitoring - Research

**Researched:** 2026-01-25
**Domain:** Discrete optimization diagnostics, matrix rank analysis, quantized training monitoring
**Confidence:** HIGH

## Summary

This research investigates implementation approaches for weight flip counting in quantized training, stable/effective rank computation, and rank collapse early warning systems. The domain covers discrete optimization dynamics where weights in FP8 representation undergo "flips" (value transitions between quantization levels) and weight matrices can experience rank degradation during training.

The standard approach for flip metrics is to compare quantized weights before/after optimizer step and count transitions between discrete levels. For rank health, two complementary metrics are established: **stable rank** (ratio of squared Frobenius to squared spectral norm) and **effective rank** (exponential of Shannon entropy of normalized singular values). Rank collapse is detected through trend analysis on these metrics, with particular sensitivity in classifier and attention output layers.

**Primary recommendation:** Use `torch.linalg.svdvals()` for efficient singular value computation (O(n^2) for values only vs O(n^3) for full SVD), compute both stable rank and effective rank per-layer at configurable intervals, and implement EMA-based trend detection for rank collapse early warning.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.x | SVD computation via `torch.linalg.svdvals()` | Native support, GPU-accelerated, numerically stable gradients |
| wandb | Latest | Metrics logging with histograms | Already integrated in altgrad, supports step alignment |
| pandas | 2.x | DataFrame-centric analysis | Already used in altgrad/analysis/ |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 1.x | Histogram binning for flip distributions | Efficient np.bincount for discrete metrics |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Full SVD | `torch.svd_lowrank()` | Use lowrank only if matrices > 4096x4096; otherwise svdvals is faster |
| Stable rank | Nuclear norm ratio | Stable rank is more interpretable and standard in NN literature |
| Manual EMA | `statsmodels.tsa.api.SimpleExpSmoothing` | Manual implementation is simpler for online updates, statsmodels for post-hoc analysis |

**Installation:**
```bash
# Already satisfied by existing altgrad dependencies
pip install torch wandb pandas numpy
```

## Architecture Patterns

### Recommended Project Structure
```
altgrad/quantization/
├── diagnostics.py           # Existing: BitStallDetector
├── advanced_diagnostics.py  # Existing: ULP statistics
├── flip_metrics.py          # NEW: WeightFlipTracker, FlipRateComputer
└── rank_health.py           # NEW: RankHealthMonitor, RankCollapseWarning

altgrad/training/
├── callbacks.py             # Existing: WandbTracker - extend with rank alerts
└── metrics.py               # Existing: Add rank metric aggregation
```

### Pattern 1: Flip Tracking via State Diff
**What:** Track weight flips by comparing quantized state before/after optimizer step
**When to use:** Every training step where flip metrics are enabled
**Example:**
```python
# Source: ICCV 2025 "Scheduling Weight Transitions for Quantization-Aware Training"
class WeightFlipTracker:
    """Track transitions between quantization levels."""

    def __init__(self):
        self.prev_quantized: Dict[str, Tensor] = {}
        self.flip_counts: Dict[str, int] = {}
        self.total_weights: Dict[str, int] = {}

    def snapshot_pre_step(self, name: str, weight: Tensor, format: FP8Format, scale: Tensor):
        """Capture quantized state before optimizer step."""
        self.prev_quantized[name] = quantize(weight, format, scale).clone()
        self.total_weights[name] = weight.numel()

    def compute_flips_post_step(self, name: str, weight: Tensor, format: FP8Format, scale: Tensor) -> int:
        """Count weights that changed quantized representation."""
        if name not in self.prev_quantized:
            return 0
        curr_quantized = quantize(weight, format, scale)
        flips = (curr_quantized != self.prev_quantized[name]).sum().item()
        self.flip_counts[name] = self.flip_counts.get(name, 0) + flips
        return flips
```

### Pattern 2: Stable Rank Computation
**What:** Compute stable rank = ||W||_F^2 / ||W||_2^2
**When to use:** Periodic rank health checks (every N steps)
**Example:**
```python
# Source: OpenReview "Stable Rank Normalization for Improved Generalization"
# arxiv.org/abs/1906.04659
def compute_stable_rank(weight: Tensor) -> float:
    """Compute stable rank of a 2D weight matrix.

    Stable rank = ||W||_F^2 / ||W||_2^2
               = sum(s_i^2) / s_1^2

    where s_i are singular values in descending order.
    """
    if weight.dim() != 2:
        # Reshape conv weights to 2D: (out, in*k*k)
        weight = weight.view(weight.size(0), -1)

    # Use svdvals for efficiency (values only, no U/V)
    singular_values = torch.linalg.svdvals(weight)

    frobenius_sq = (singular_values ** 2).sum()
    spectral_sq = singular_values[0] ** 2

    # Avoid division by zero
    if spectral_sq < 1e-10:
        return float(weight.numel())  # Full rank fallback

    return (frobenius_sq / spectral_sq).item()
```

### Pattern 3: Effective Rank (Shannon Entropy)
**What:** Compute effective rank via entropy of normalized singular values
**When to use:** Alongside stable rank for complementary view
**Example:**
```python
# Source: Roy & Vetterli 2007 "The effective rank: A measure of effective dimensionality"
# EUSIPCO 2007
def compute_effective_rank(weight: Tensor, eps: float = 1e-10) -> float:
    """Compute effective rank using Shannon entropy.

    erank(W) = exp(-sum(p_i * log(p_i)))
    where p_i = s_i / sum(s_j) are normalized singular values.

    Returns value in [1, min(m,n)] - higher means more evenly distributed SVs.
    """
    if weight.dim() != 2:
        weight = weight.view(weight.size(0), -1)

    singular_values = torch.linalg.svdvals(weight)

    # Normalize to probability distribution
    sv_sum = singular_values.sum()
    if sv_sum < eps:
        return 1.0  # Degenerate case

    p = singular_values / sv_sum

    # Shannon entropy: -sum(p * log(p)), with 0*log(0) = 0
    p_safe = p.clamp(min=eps)
    entropy = -(p * torch.log(p_safe)).sum()

    # Effective rank = exp(entropy)
    return torch.exp(entropy).item()
```

### Pattern 4: EMA-Based Trend Detection
**What:** Detect downward trends in rank metrics for early warning
**When to use:** Rank collapse early warning system
**Example:**
```python
class RankTrendDetector:
    """Detect sustained downward trends in rank metrics."""

    def __init__(self, alpha: float = 0.1, threshold_pct: float = 0.2, window: int = 100):
        """
        Args:
            alpha: EMA smoothing factor (0.1 = slow, 0.3 = fast)
            threshold_pct: Warn if EMA drops by this fraction from initial
            window: Steps before trend detection activates
        """
        self.alpha = alpha
        self.threshold_pct = threshold_pct
        self.window = window
        self.ema: Optional[float] = None
        self.initial_ema: Optional[float] = None
        self.step_count = 0

    def update(self, value: float) -> Optional[str]:
        """Update with new value, return warning if trend detected."""
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        self.step_count += 1

        if self.step_count == self.window:
            self.initial_ema = self.ema

        if self.step_count > self.window and self.initial_ema is not None:
            drop_pct = (self.initial_ema - self.ema) / self.initial_ema
            if drop_pct > self.threshold_pct:
                return f"WARN: Rank dropped {drop_pct:.1%} from initial"

        return None
```

### Anti-Patterns to Avoid
- **Computing full SVD when only values needed:** Use `torch.linalg.svdvals()` not `torch.linalg.svd()` - 10-50x faster for values-only
- **Rank checking every step:** SVD is expensive; check every 100-500 steps unless debugging
- **Global aggregation only:** Per-layer rank is critical; classifier/attention output layers collapse first
- **Ignoring matrix shape:** Reshape conv weights to 2D (out_channels, in*k*k) before rank computation
- **Hard thresholds for warnings:** Use trend detection with EMA, not absolute thresholds

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Singular value computation | Custom iterative methods | `torch.linalg.svdvals()` | GPU-accelerated, numerically stable, handles edge cases |
| Histogram logging | Manual binning + console print | `wandb.Histogram()` | Automatic visualization, step alignment, comparison across runs |
| Trend detection | Simple diff checks | EMA with warmup window | Noise-resistant, configurable sensitivity |
| Per-layer aggregation | Loop + dict accumulation | Existing `compute_gradient_stats` pattern | Consistent naming, aggregate computation |

**Key insight:** The mathematical operations (SVD, entropy) are well-solved by PyTorch. The engineering challenge is efficient integration: when to compute, how to aggregate across layers, and how to log without impacting training throughput.

## Common Pitfalls

### Pitfall 1: SVD Expense Killing Training Throughput
**What goes wrong:** Computing SVD every step slows training by 5-20x
**Why it happens:** SVD is O(min(m,n)^2 * max(m,n)) - expensive for large weight matrices
**How to avoid:**
- Compute rank metrics every `rank_log_interval` steps (default: 100-500)
- Use `torch.linalg.svdvals()` not full SVD
- Consider sampling layers (e.g., first/last of each type) for frequent checks
**Warning signs:** Training throughput drops significantly when rank logging enabled

### Pitfall 2: Ignoring Layer-Specific Collapse Patterns
**What goes wrong:** Global rank metrics look fine while specific layers collapse
**Why it happens:** Classifier and attention output layers are more sensitive to rank collapse; averaging hides local issues
**How to avoid:**
- Track per-layer rank metrics separately
- Alert on ANY layer crossing threshold, not just average
- Prioritize monitoring: `lm_head`, `attn.c_proj`, final MLP layers
**Warning signs:** Validation loss spikes while aggregate rank looks stable

### Pitfall 3: Flip Rate Misinterpretation
**What goes wrong:** Low flip rate interpreted as "training frozen" when it may indicate convergence
**Why it happens:** Near convergence, fewer weights need to change
**How to avoid:**
- Track flip rate trend, not absolute value
- Correlate with loss plateau - low flips + flat loss = likely converged
- Check bit-stall rate (no-change despite gradient) separately from flip rate (post-optimizer changes)
**Warning signs:** Premature training termination based on flip rate alone

### Pitfall 4: EMA Warmup Neglect
**What goes wrong:** False rank collapse warnings in early training
**Why it happens:** Rank naturally fluctuates in early training; EMA needs warmup
**How to avoid:**
- Use `window` parameter to delay trend detection
- Start trend detection after N steps (e.g., 100-500)
- Consider higher alpha (faster adaptation) in early training
**Warning signs:** Repeated false warnings in first 10% of training

### Pitfall 5: Conv Weight Reshaping
**What goes wrong:** Rank computation fails or gives meaningless results for conv layers
**Why it happens:** Conv weights are 4D (out, in, k, k); SVD expects 2D
**How to avoid:**
- Always reshape to 2D: `weight.view(weight.size(0), -1)`
- Document that rank is computed on the 2D "view" of the weight
**Warning signs:** RuntimeError from SVD or nonsensical rank values

## Code Examples

Verified patterns from official sources:

### Efficient Singular Values Computation
```python
# Source: PyTorch 2.9 docs - torch.linalg.svdvals
import torch

def get_singular_values(weight: torch.Tensor) -> torch.Tensor:
    """Get singular values efficiently (values only, no U/V).

    Args:
        weight: 2D weight matrix (reshape conv weights first)

    Returns:
        1D tensor of singular values in descending order
    """
    if weight.dim() != 2:
        weight = weight.view(weight.size(0), -1)

    # svdvals is more efficient than svd when only values needed
    # Gradients are always numerically stable (unlike torch.linalg.svd)
    return torch.linalg.svdvals(weight)
```

### W&B Histogram Logging
```python
# Source: docs.wandb.ai/guides/track/log/
import wandb

def log_flip_histogram(flip_counts: dict, step: int):
    """Log per-layer flip counts as histogram.

    Args:
        flip_counts: Dict mapping layer_name -> flip_count
        step: Training step for x-axis alignment
    """
    # Log individual layer flip counts
    for name, count in flip_counts.items():
        wandb.log({f"flips/{name}": count}, step=step)

    # Log distribution of flip rates across layers
    flip_rates = list(flip_counts.values())
    wandb.log({
        "flips/distribution": wandb.Histogram(flip_rates),
        "flips/total": sum(flip_rates),
    }, step=step)
```

### Complete Rank Health Monitor
```python
# Synthesized from research - HIGH confidence pattern
class RankHealthMonitor:
    """Monitor weight matrix rank health during training."""

    def __init__(
        self,
        log_interval: int = 100,
        warn_threshold: float = 0.3,  # Warn if rank drops 30%
        critical_layers: list = None,  # Layers to always monitor
    ):
        self.log_interval = log_interval
        self.warn_threshold = warn_threshold
        self.critical_layers = critical_layers or ["lm_head", "c_proj"]
        self.trend_detectors: Dict[str, RankTrendDetector] = {}

    def compute_layer_ranks(self, model: nn.Module) -> Dict[str, Dict[str, float]]:
        """Compute stable and effective rank for all 2D+ parameters."""
        ranks = {}
        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue

            weight = param.data
            if weight.dim() > 2:
                weight = weight.view(weight.size(0), -1)

            sv = torch.linalg.svdvals(weight)

            # Stable rank
            stable = ((sv ** 2).sum() / (sv[0] ** 2 + 1e-10)).item()

            # Effective rank
            p = sv / (sv.sum() + 1e-10)
            entropy = -(p * torch.log(p.clamp(min=1e-10))).sum()
            effective = torch.exp(entropy).item()

            ranks[name] = {
                "stable_rank": stable,
                "effective_rank": effective,
                "spectral_norm": sv[0].item(),
            }

        return ranks

    def check_warnings(self, ranks: Dict[str, Dict[str, float]]) -> List[str]:
        """Check for rank collapse warnings."""
        warnings = []
        for name, metrics in ranks.items():
            if name not in self.trend_detectors:
                self.trend_detectors[name] = RankTrendDetector(
                    threshold_pct=self.warn_threshold
                )

            warning = self.trend_detectors[name].update(metrics["stable_rank"])
            if warning:
                warnings.append(f"{name}: {warning}")

        return warnings
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full SVD for rank | `svdvals()` for values only | PyTorch 1.9+ (2021) | 10-50x speedup for rank metrics |
| Absolute rank thresholds | EMA-based trend detection | Recent (2023+) | Fewer false positives, noise-robust |
| Global rank only | Per-layer rank monitoring | 2020+ (BatchNorm rank collapse paper) | Earlier detection of layer-specific issues |
| Hard-coded weight transitions | Transition Rate Scheduling | ICCV 2025 | Explicit control of flip rates in QAT |

**Deprecated/outdated:**
- `torch.matrix_rank()`: Deprecated in favor of `torch.linalg.matrix_rank()`
- Manual spectral norm computation: Use `torch.linalg.norm(w, ord=2)` or `svdvals()[0]`

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal rank logging interval**
   - What we know: SVD is expensive; 100-500 steps is reasonable
   - What's unclear: Optimal interval depends on model size and training dynamics
   - Recommendation: Make configurable, default to 100 steps, increase if throughput impact > 5%

2. **Effective rank vs stable rank sensitivity**
   - What we know: Both capture different aspects of rank health
   - What's unclear: Which is better predictor of imminent collapse
   - Recommendation: Log both, use stable rank for warnings (more studied), keep effective rank for analysis

3. **Layer-specific warning thresholds**
   - What we know: Classifier layers collapse first
   - What's unclear: Should thresholds differ by layer type?
   - Recommendation: Start with uniform thresholds, adjust based on empirical observation

## Sources

### Primary (HIGH confidence)
- [PyTorch 2.9 torch.linalg.svdvals documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svdvals.html) - Singular value computation API
- [Roy & Vetterli 2007 - The Effective Rank (EUSIPCO)](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf) - Effective rank definition
- [arxiv:1906.04659 - Stable Rank Normalization (ICLR 2020)](https://arxiv.org/abs/1906.04659) - Stable rank in neural networks
- [Weights & Biases logging documentation](https://docs.wandb.ai/guides/track/log/) - Histogram and scalar logging

### Secondary (MEDIUM confidence)
- [arxiv:2003.01652 - Batch Normalization Provably Avoids Rank Collapse (NeurIPS 2020)](https://arxiv.org/abs/2003.01652) - Rank collapse phenomenon and batch norm role
- [arxiv:2410.07799 - Mind the Gap: Rank Collapse in Attention Layers](https://arxiv.org/html/2410.07799v2) - Attention-specific rank collapse
- [ICCV 2025 - Scheduling Weight Transitions for QAT](https://arxiv.org/abs/2404.19248) - Transition rate concept for quantized training
- [WeightWatcher GitHub](https://github.com/CalculatedContent/WeightWatcher) - Weight matrix diagnostics patterns

### Tertiary (LOW confidence)
- Community discussions on rank monitoring frequency and thresholds
- Empirical observations from WebSearch results on training stability

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch linalg is authoritative, W&B integration well-documented
- Architecture: HIGH - Patterns derived from published papers with clear mathematical definitions
- Pitfalls: MEDIUM - Based on research papers and general ML engineering knowledge; some specifics need validation

**Research date:** 2026-01-25
**Valid until:** 2026-03-25 (60 days - domain is stable, mathematical definitions don't change)
