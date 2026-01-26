# Phase 8: Update Metrics & Test Matrix - Research

**Researched:** 2026-01-26
**Domain:** Discrete optimization metrics, grid-based FP8 optimizer, test documentation
**Confidence:** HIGH

## Summary

This phase implements granular metrics for discrete optimization (distinguishing attempted updates from successful FP8 flips), a grid-based optimizer following the user-provided MasterOptim reference pattern, classifier-specific rank monitoring, and a comprehensive test matrix document.

The codebase already has foundational components from Phase 7: `WeightFlipTracker` in `flip_metrics.py` counts flips, and `RankHealthMonitor` in `rank_health.py` tracks classifier layers with `critical_layers=["lm_head", "c_proj"]`. Phase 8 extends these by:
1. Adding **update counting** (non-zero gradient applied) distinct from flip counting
2. Computing **stall ratio** = 1 - (flips / updates) to quantify gradient effectiveness
3. Implementing **GridOptim** following MasterOptim patterns: FP32 master weights, explicit grid from FP8 representable values, stochastic rounding, rung clipping
4. Enhancing classifier rank monitoring with stricter thresholds for lm_head/c_proj
5. Creating **TEST_MATRIX.md** documenting all format x optimizer x layer combinations

**Primary recommendation:** Implement GridOptim as a new optimizer class in `altgrad/training/optimizer.py`, extend `WeightFlipTracker` to track updates alongside flips, and add a `TEST_MATRIX.md` artifact at the project root.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x | Core tensor operations, optimizer base | Project foundation |
| torch.searchsorted | 2.x | Grid index lookup for rung-based updates | Native vectorized binary search |
| torch.floor + rand_like | 2.x | Stochastic rounding implementation | Efficient random + floor pattern |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.arange | 2.x | Build explicit FP8 grid | Initial grid construction |
| torch.clamp | 2.x | Rung clipping to avoid NaN cliffs | Every optimizer step |
| torch.unique + sort | 2.x | Clean grid (remove NaN/Inf) | Grid initialization |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom grid lookup | torch.bucketize | searchsorted is more flexible for sorted sequences |
| Manual stochastic round | QPyTorch | QPyTorch is heavier dependency, custom is simpler |
| Direct FP8 dtype | Simulated FP8 in FP32 | Native FP8 requires CUDA 11.8+ and specific hardware |

**Installation:**
No additional packages required - all functionality available in base PyTorch.

## Architecture Patterns

### Recommended Project Structure
```
altgrad/
├── quantization/
│   ├── flip_metrics.py      # Extended: add update tracking
│   └── rank_health.py       # Extended: stricter classifier thresholds
├── training/
│   └── optimizer.py         # New: GridOptim class alongside ManifoldAdamW
└── TEST_MATRIX.md           # New: experiment documentation
```

### Pattern 1: Master Weight + Grid-Based Discrete Updates
**What:** Maintain FP32 master weights, project to FP8 grid positions, use stochastic rounding in rung space
**When to use:** Training with discrete FP8 weights where bit-level granularity matters
**Example:**
```python
# Source: User-provided reference_optimizer.py
class GridOptim:
    def __init__(self, params, scale=6.0, momentum=0.9, weight_decay=1e-4):
        self.params = list(params)
        self.scale = scale
        self.momentum = momentum
        self.wd = weight_decay

        # FP32 master weights (copy from FP8 model weights)
        self.master_p = [p.detach().clone().float() for p in self.params]
        self.velocity = [torch.zeros_like(p).float() for p in self.master_p]

        # Build explicit FP8 grid from all representable values
        raw_bits = torch.arange(-128, 128, dtype=torch.int8)
        all_floats = raw_bits.view(DTYPE).to(torch.float32)  # Reinterpret as FP8
        clean_grid = all_floats[~torch.isnan(all_floats) & ~torch.isinf(all_floats)]
        self.grid = torch.sort(torch.unique(clean_grid))[0]
```

### Pattern 2: Stochastic Rounding in Rung Space
**What:** Convert continuous updates to discrete rung movements using probabilistic rounding
**When to use:** Every grid-based optimizer step
**Example:**
```python
# Source: User-provided reference_optimizer.py
@torch.no_grad()
def step(self, current_scale=None):
    flips = 0
    updates = 0  # NEW: track attempted updates
    scale = current_scale if current_scale is not None else self.scale

    for i, p in enumerate(self.params):
        if p.grad is None:
            continue
        old_data = p.data.clone()
        grad = p.grad.to(torch.float32)

        # Count updates (non-zero gradient)
        updates += (grad.abs() > 1e-10).sum().item()  # NEW

        if self.wd != 0:
            grad.add_(self.master_p[i], alpha=self.wd)

        self.velocity[i] = self.momentum * self.velocity[i] + grad

        # Find current rung indices
        indices = torch.searchsorted(self.grid, self.master_p[i].contiguous())

        # Scale velocity to rung units
        v_rungs = self.velocity[i] * scale

        # CRITICAL: Clip to avoid NaN cliffs at grid boundaries
        v_rungs = torch.clamp(v_rungs, -10, 10)

        # Stochastic rounding: floor(x + rand) gives probabilistic round
        v_rounded = torch.floor(v_rungs + torch.rand_like(v_rungs)).to(torch.int32)

        # Move to new rung position (subtract because gradient descent)
        new_indices = torch.clamp(indices - v_rounded, 0, len(self.grid) - 1)
        new_floats = self.grid[new_indices.long()].view(p.shape)

        # Update master and model weights
        self.master_p[i].copy_(new_floats)
        p.data.copy_(new_floats.to(DTYPE))

        # Count actual FP8 changes (flips)
        flips += (p.data != old_data).sum().item()

    return flips, updates  # NEW: return both counts
```

### Pattern 3: Update vs Flip Distinction
**What:** Separate "attempted update" (non-zero gradient) from "successful flip" (FP8 representation changed)
**When to use:** Diagnosing gradient effectiveness in discrete training
**Example:**
```python
# Source: Extending existing WeightFlipTracker
class UpdateFlipTracker:
    """Track both update attempts and successful flips."""

    def __init__(self):
        self.flip_counts: Dict[str, int] = {}
        self.update_counts: Dict[str, int] = {}  # NEW
        self.total_weights: Dict[str, int] = {}

    def record(self, name: str, flips: int, updates: int, total: int):
        """Record metrics for a layer."""
        self.flip_counts[name] = self.flip_counts.get(name, 0) + flips
        self.update_counts[name] = self.update_counts.get(name, 0) + updates
        if name not in self.total_weights:
            self.total_weights[name] = total

    def get_stall_ratio(self, name: str) -> float:
        """Stall ratio = 1 - (flips / updates).

        0 = all updates cause flips (ideal)
        1 = no updates cause flips (complete stall)
        """
        updates = self.update_counts.get(name, 0)
        flips = self.flip_counts.get(name, 0)
        if updates == 0:
            return 0.0  # No updates attempted
        return 1.0 - (flips / updates)
```

### Pattern 4: Classifier-Specific Stricter Monitoring
**What:** Apply tighter rank degradation thresholds to critical classifier layers
**When to use:** Monitoring lm_head, c_proj, and other output-critical layers
**Example:**
```python
# Source: Extending existing RankHealthMonitor
class RankHealthMonitor:
    def __init__(
        self,
        log_interval: int = 100,
        warn_threshold: float = 0.3,
        critical_layers: Optional[List[str]] = None,
        critical_threshold_multiplier: float = 0.5,  # NEW: stricter for classifiers
    ):
        self.critical_layers = critical_layers or ["lm_head", "c_proj"]
        self.warn_threshold = warn_threshold
        self.critical_multiplier = critical_threshold_multiplier

    def _get_threshold_for_layer(self, name: str) -> float:
        """Return stricter threshold for critical layers."""
        is_critical = any(pattern in name for pattern in self.critical_layers)
        if is_critical:
            # E.g., 0.3 * 0.5 = 0.15 for classifiers vs 0.3 for others
            return self.warn_threshold * self.critical_multiplier
        return self.warn_threshold
```

### Anti-Patterns to Avoid
- **Rounding without clipping:** NEVER apply stochastic rounding without clamping v_rungs first. Large jumps can land on NaN/Inf grid positions.
- **Counting flips as updates:** They are distinct metrics. Flip counts how many FP8 values changed; update counts how many received non-zero gradients.
- **Using FP8 master weights:** Master weights must be FP32 to accumulate small gradients over time. Only project to FP8 at the end of each step.
- **Ignoring grid boundaries:** Always use `torch.clamp(indices, 0, len(grid) - 1)` after adding rung offsets.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Binary search for grid index | Manual loop over grid | torch.searchsorted | Vectorized, GPU-optimized |
| Stochastic rounding | Custom probability code | floor(x + rand_like(x)) | Standard pattern, verified correct |
| FP8 representable values | Computing from formula | torch.arange(-128,128).view(DTYPE) | Exact bit patterns, no formula errors |
| Rank computation | Manual SVD extraction | torch.linalg.svdvals | Numerically stable, fast |
| Layer name matching | Exact string equality | `any(pattern in name for ...)` | Handles nested modules, suffixes |

**Key insight:** The grid-based optimization pattern from the reference is elegant but has subtle correctness requirements. The explicit grid construction via `arange().view(DTYPE)` captures exactly the representable values without relying on format-specific formulas that could have edge case bugs.

## Common Pitfalls

### Pitfall 1: NaN from Grid Boundary Overflow
**What goes wrong:** Stochastic rounding produces index outside valid grid range, indexing returns NaN/Inf
**Why it happens:** Large velocity * scale can produce rung movement > grid size
**How to avoid:** Always clamp v_rungs before rounding: `torch.clamp(v_rungs, -10, 10)`
**Warning signs:** Random NaN appearance in weights after optimizer step

### Pitfall 2: Integer Overflow in Rung Indices
**What goes wrong:** v_rounded overflows int32 range, wraps to negative, produces wrong indices
**Why it happens:** Extremely large gradients or scale factors
**How to avoid:** Cast to int32 AFTER clamping, not before
**Warning signs:** Sudden weight jumps to unexpected values

### Pitfall 3: Confusing Stall Ratio Direction
**What goes wrong:** Misinterpreting stall_ratio: 0 is good (all updates flip), 1 is bad (no flips)
**Why it happens:** Natural assumption that higher = better
**How to avoid:** Document clearly: `stall_ratio = 1 - (flips / updates)`
**Warning signs:** Celebrating high stall ratios as "stable training"

### Pitfall 4: Non-Contiguous Tensor in searchsorted
**What goes wrong:** searchsorted returns incorrect indices for non-contiguous tensors
**Why it happens:** PyTorch searchsorted requires contiguous memory layout
**How to avoid:** Always call `.contiguous()` on the values tensor
**Warning signs:** Off-by-one or wildly wrong grid indices

### Pitfall 5: Classifier Rank Degradation Masked by Averaging
**What goes wrong:** Critical lm_head rank collapse hidden in mean rank statistics
**Why it happens:** Mean across all layers dilutes single-layer problems
**How to avoid:** Track per-layer metrics AND apply stricter thresholds to critical layers
**Warning signs:** Good average rank but model outputs collapse

## Code Examples

Verified patterns from official sources and reference implementation:

### Building FP8 Grid
```python
# Source: reference_optimizer.py (user-provided)
DTYPE = torch.float8_e4m3fn  # Or other FP8 dtype

# Build explicit grid of all representable FP8 values
raw_bits = torch.arange(-128, 128, dtype=torch.int8)
all_floats = raw_bits.view(DTYPE).to(torch.float32)

# Remove NaN and Inf
clean_grid = all_floats[~torch.isnan(all_floats) & ~torch.isinf(all_floats)]
grid = torch.sort(torch.unique(clean_grid))[0].to(device)
```

### Stochastic Rounding Step
```python
# Source: reference_optimizer.py (user-provided)
# Find current position in grid
indices = torch.searchsorted(self.grid, master_weights.contiguous())

# Convert velocity to rung units and clip
v_rungs = velocity * scale
v_rungs = torch.clamp(v_rungs, -10, 10)  # CRITICAL: prevent NaN

# Stochastic round: probability of rounding up = fractional part
v_rounded = torch.floor(v_rungs + torch.rand_like(v_rungs)).to(torch.int32)

# Move indices (subtract for descent)
new_indices = torch.clamp(indices - v_rounded, 0, len(self.grid) - 1)
new_weights = self.grid[new_indices.long()]
```

### Flip Counting
```python
# Source: reference_optimizer.py (user-provided)
old_data = param.data.clone()
# ... optimizer step ...
param.data.copy_(new_floats.to(DTYPE))

# Count element-wise changes
flips = (param.data != old_data).sum().item()
```

### Update Counting
```python
# Count non-zero gradients (attempted updates)
updates = (grad.abs() > 1e-10).sum().item()
```

### Stall Ratio Computation
```python
def compute_stall_ratio(flips: int, updates: int) -> float:
    """Stall ratio = 1 - (flips / updates).

    Returns:
        0.0: All updates caused flips (ideal)
        1.0: No updates caused flips (complete stall)
        0.0 if updates == 0 (no gradient)
    """
    if updates == 0:
        return 0.0
    return 1.0 - (flips / updates)
```

### TEST_MATRIX.md Structure
```markdown
# Test Matrix

## Formats
| Format | Exponent | Mantissa | Range | Notes |
|--------|----------|----------|-------|-------|
| E0M7 | 0 | 7 | [-1, 1) | Fixed-point |
| E1M6 | 1 | 6 | Two scales | |
| E3M4 | 3 | 4 | ~0.06-124 | Moderate range |
| E5M2 | 5 | 2 | Wide | Standard FP8 |
| E7M0 | 7 | 0 | Powers of 2 | Extreme |

## Optimizers
| Optimizer | Master Weights | Update Mechanism | Key Parameters |
|-----------|---------------|------------------|----------------|
| ManifoldAdamW | Implicit FP32 | Stiffness-preconditioned | mantissa_bits, max_stiffness |
| GridOptim | Explicit FP32 | Rung-based stochastic | scale, rung_clip |
| AdamW | FP32 | Standard | lr, betas, weight_decay |

## Layer Types
| Layer Type | Critical | Threshold | Monitoring |
|------------|----------|-----------|------------|
| lm_head | Yes | 0.15 | Rank collapse early warning |
| c_proj | Yes | 0.15 | Rank collapse early warning |
| Others | No | 0.30 | Standard monitoring |

## Test Combinations
| Format | Optimizer | Layer Focus | Status | Notes |
|--------|-----------|-------------|--------|-------|
| E5M2 | ManifoldAdamW | All | Baseline | Standard comparison |
| E5M2 | GridOptim | All | Primary | Grid vs stiffness |
| E3M4 | GridOptim | All | Test | Higher precision |
| E7M0 | GridOptim | All | Extreme | Powers of 2 only |
| ... | ... | ... | ... | ... |
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Bit-stall only | Update + Flip + Stall Ratio | Phase 8 | Granular gradient effectiveness |
| Fixed clip threshold | Rung-based clipping (-10, 10) | MasterOptim pattern | Prevents NaN from grid overflow |
| Uniform rank thresholds | Layer-specific thresholds | Phase 8 | Earlier classifier collapse detection |
| Implicit precision loss | Explicit grid-based updates | MasterOptim pattern | Direct control over FP8 transitions |

**Deprecated/outdated:**
- Simple flip rate without update context: Now requires computing stall_ratio for meaningful interpretation
- Using FP8 directly for master weights: FP32 master weights are essential for gradient accumulation

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal rung_clip value**
   - What we know: Reference uses 10, prevents NaN at grid boundaries
   - What's unclear: Whether 10 is optimal for all formats (E7M0 has fewer rungs)
   - Recommendation: Start with 10, expose as parameter, tune per format

2. **Stochastic rounding vs deterministic for inference**
   - What we know: Stochastic during training preserves expected value
   - What's unclear: Whether to switch to deterministic at eval time
   - Recommendation: Use stochastic during training, deterministic at inference

3. **Grid construction for custom formats**
   - What we know: torch.arange().view(DTYPE) works for standard FP8 dtypes
   - What's unclear: How to build grid for simulated formats (E0M7, E1M6) without native dtype
   - Recommendation: Use existing `FP8Format.to_real()` to enumerate all 256 bit patterns

## Sources

### Primary (HIGH confidence)
- User-provided `reference_optimizer.py` - Complete MasterOptim implementation with grid-based updates
- Existing codebase `altgrad/quantization/flip_metrics.py` - WeightFlipTracker pattern
- Existing codebase `altgrad/quantization/rank_health.py` - RankHealthMonitor with critical_layers
- [PyTorch torch.searchsorted documentation](https://docs.pytorch.org/docs/stable/generated/torch.searchsorted.html) - Binary search API

### Secondary (MEDIUM confidence)
- [TorchAO quantized training](https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training) - Stochastic rounding patterns
- [Direct Quantized Training with Stochastic Rounding](https://arxiv.org/html/2412.04787v1) - DQT methodology
- [FP8-LM Training Best Practices](https://arxiv.org/html/2310.18313v2) - Master weights in FP32

### Tertiary (LOW confidence)
- WebSearch results on experiment documentation - TEST_MATRIX format patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using only PyTorch built-ins, reference implementation verified
- Architecture: HIGH - User-provided MasterOptim is working reference, extending existing codebase patterns
- Pitfalls: HIGH - Reference implementation explicitly addresses NaN cliff issue with clipping

**Research date:** 2026-01-26
**Valid until:** 60 days (stable PyTorch APIs, custom project patterns)
