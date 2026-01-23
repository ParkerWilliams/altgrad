# Phase 5: Manifold-Aware Optimizer - Research

**Researched:** 2026-01-22
**Domain:** Custom PyTorch optimizer with stiffness-preconditioned gradient updates for FP8 quantization
**Confidence:** HIGH

## Summary

This phase implements a manifold-aware optimizer that treats the FP8 quantization grid as a geometric manifold with non-uniform spacing. The core insight is that standard optimizers apply uniform real-valued updates, but the FP8 grid has exponentially varying ULP (unit in last place) sizes. A stiffness-preconditioned update scales gradients by the local grid spacing so that updates move weights by a consistent number of ULPs rather than a fixed real-valued delta.

The research confirms that PyTorch's optimizer base class provides robust infrastructure for custom optimizers: subclass `torch.optim.Optimizer`, implement `step()` with `@torch.no_grad()`, and use `self.state[p]` for per-parameter state (momentum, bit-position tracking). The existing `compute_stiffness_field()` in `advanced_diagnostics.py` already implements the stiffness formula S = 2^(floor(log2|w|) - M), which can be reused directly.

**Primary recommendation:** Implement a `ManifoldAdamW` optimizer that wraps the standard AdamW computation and multiplies gradients by stiffness before the Adam moment updates. Maintain a "bit-position" tensor in per-parameter state to track the latent integer representation. Use a toggle parameter to switch between standard and manifold-aware modes for controlled comparison.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9+ | Optimizer base class | `torch.optim.Optimizer` provides state management, param_groups, checkpointing |
| altgrad.quantization | internal | Stiffness computation | `compute_stiffness_field()` already implements MANI-01 formula |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.nextafter | PyTorch 2.x | IEEE 754 ULP computation | Already used in `compute_ulp_distance()` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom optimizer | Optimizer wrapper/hook | Wrapper adds complexity; direct subclass is cleaner for state management |
| Pre-step gradient modification | `register_step_pre_hook` | Hook approach less explicit; direct implementation clearer for research |
| Stiffness in optimizer | Stiffness in QuantizedLinear backward | Optimizer-level cleaner separation of concerns |

**Installation:**
```bash
# No additional dependencies - uses existing PyTorch and altgrad modules
```

## Architecture Patterns

### Recommended Project Structure
```
altgrad/
├── quantization/
│   ├── advanced_diagnostics.py   # compute_stiffness_field() - REUSE
│   └── diagnostics.py            # BitStallDetector - REUSE
├── training/
│   ├── optimizer.py              # NEW: ManifoldAdamW
│   ├── trainer.py                # UPDATE: support manifold optimizer
│   └── config.py                 # UPDATE: add manifold config options
└── tests/
    └── test_optimizer.py         # NEW: optimizer unit tests
```

### Pattern 1: Custom Optimizer with Per-Parameter State

**What:** PyTorch custom optimizer with state dictionary for tracking per-parameter values
**When to use:** Any optimizer that needs to maintain history per parameter (momentum, bit-position)

```python
# Source: PyTorch torch.optim.Optimizer pattern
import torch
from torch.optim import Optimizer

class ManifoldAdamW(Optimizer):
    """AdamW with stiffness-preconditioned gradient updates."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        manifold_aware: bool = True,
        mantissa_bits: int = 2,  # E5M2 default
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
            manifold_aware=manifold_aware,
            mantissa_bits=mantissa_bits,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Access per-parameter state
                state = self.state[p]

                # Initialize state on first call
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Bit-position tracking (MANI-04)
                    state["bit_position"] = self._init_bit_position(p)

                # ... update logic here ...

        return loss
```

**Key points:**
- `@torch.no_grad()` decorator prevents gradient tracking during updates
- `self.state[p]` is a dict keyed by parameter tensor
- State initialized lazily on first step
- `closure` pattern supports optimizers that need loss re-evaluation

### Pattern 2: Stiffness-Preconditioned Gradient Update

**What:** Multiply gradients by stiffness factor before Adam moment updates
**When to use:** MANI-02 requirement

```python
# Source: Derived from project description and compute_stiffness_field
from altgrad.quantization.advanced_diagnostics import compute_stiffness_field

def _precondition_gradient(self, grad: torch.Tensor, weight: torch.Tensor, mantissa_bits: int) -> torch.Tensor:
    """Apply stiffness preconditioning to gradient (MANI-02).

    Multiplies gradient by local stiffness factor so that the
    effective update size is measured in ULPs, not real values.

    Args:
        grad: Gradient tensor
        weight: Current weight tensor
        mantissa_bits: Format mantissa bits (M in S formula)

    Returns:
        Preconditioned gradient
    """
    # Compute stiffness: S = 2^(floor(log2|w|) - M)
    stiffness = compute_stiffness_field(weight, mantissa_bits)

    # Handle NaN stiffness (zero weights)
    stiffness = torch.where(
        torch.isnan(stiffness),
        torch.ones_like(stiffness),  # Use 1.0 for zero weights
        stiffness
    )

    # Precondition: grad_effective = grad * S
    return grad * stiffness
```

**Key points:**
- Stiffness field already implemented in Phase 4
- NaN handling for zero weights (stiffness undefined)
- Multiplication scales gradient by local grid spacing

### Pattern 3: Bit-Position Tracking (Latent Integer State)

**What:** Track weights as latent integers in the quantized representation
**When to use:** MANI-04 requirement

```python
# Source: Derived from project description
def _init_bit_position(self, weight: torch.Tensor) -> torch.Tensor:
    """Initialize bit-position state from current weights.

    The bit-position represents the weight as an integer index
    into the FP8 grid. This allows tracking how many ULPs the
    weight moves per update.

    For simplicity, we track the fractional ULP position as a float
    rather than actual integer indices.
    """
    # Initialize to 0.0 - relative position within current ULP
    return torch.zeros_like(weight)

def _update_bit_position(
    self,
    state: dict,
    weight_before: torch.Tensor,
    weight_after: torch.Tensor,
) -> None:
    """Update bit-position state after weight update.

    Tracks cumulative ULP movement for analysis.
    """
    # ULP at each position
    inf_tensor = torch.full_like(weight_before, float("inf"))
    ulp = torch.abs(torch.nextafter(weight_before, inf_tensor) - weight_before)
    safe_ulp = ulp.clamp(min=1e-45)

    # Movement in ULPs
    delta_ulps = (weight_after - weight_before) / safe_ulp

    # Accumulate (or track running average)
    state["bit_position"] += delta_ulps
```

### Pattern 4: Standard vs Manifold-Aware Toggle

**What:** Runtime toggle between standard and manifold-aware modes
**When to use:** MANI-03 requirement

```python
# In step() function:
if group["manifold_aware"]:
    # Precondition gradient by stiffness
    grad = self._precondition_gradient(grad, p.data, group["mantissa_bits"])

# Then proceed with standard AdamW update
# exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
# exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
# ...
```

### Pattern 5: State Dict for Checkpointing

**What:** Automatic checkpointing through standard state_dict() interface
**When to use:** Resume training with optimizer state

```python
# Source: PyTorch torch.optim.Optimizer
# state_dict() and load_state_dict() inherited from base class

# Save checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),  # Includes bit_position
    "step": step,
}
torch.save(checkpoint, "checkpoint.pt")

# Load checkpoint
checkpoint = torch.load("checkpoint.pt")
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# bit_position and other state automatically restored
```

### Anti-Patterns to Avoid

- **Modifying gradients in-place before backward complete:** Always use `.clone()` or operate on copies to avoid autograd errors
- **Forgetting `@torch.no_grad()`:** Step function must not track gradients of the update itself
- **State key collision:** Use distinct keys for each state variable (step, exp_avg, bit_position)
- **Hard-coded mantissa bits:** Accept format via parameter, don't assume E5M2
- **Ignoring weight_decay:** Apply weight decay correctly (AdamW applies to params, not gradients)
- **Stiffness with zero weights:** Handle NaN stiffness from zero weights (use 1.0 or skip)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Optimizer base class | Custom state management | `torch.optim.Optimizer` | Handles param_groups, state_dict, hooks |
| Stiffness formula | New stiffness function | `compute_stiffness_field()` | Already tested in Phase 4 |
| ULP computation | Manual IEEE 754 math | `torch.nextafter` | IEEE 754 compliant, handles edge cases |
| Adam moment tracking | Manual exp_avg | Study PyTorch Adam source | Bias correction, numerical stability |
| Weight decay | Gradient modification | AdamW decoupled weight decay | Correct formula is `param.mul_(1 - lr * wd)` |
| Checkpoint serialization | Custom save format | `state_dict()`/`load_state_dict()` | Automatic tensor serialization |

**Key insight:** PyTorch's optimizer infrastructure handles the hard parts (state management, checkpointing, parameter groups). Focus the implementation on the stiffness preconditioning logic, not optimizer plumbing.

## Common Pitfalls

### Pitfall 1: Stiffness Explosion at Large Magnitudes

**What goes wrong:** Stiffness grows exponentially with weight magnitude; can cause massive effective gradients
**Why it happens:** S = 2^(floor(log2|w|) - M) grows exponentially; E7M0 has M=0, worst case
**How to avoid:** Clip stiffness to maximum value, or normalize by median stiffness
**Warning signs:** NaN loss after manifold-aware update, exploding gradient norms

**Prevention strategy:**
```python
# Clip stiffness to prevent explosion
max_stiffness = 1e6  # Configurable
stiffness = stiffness.clamp(max=max_stiffness)
```

### Pitfall 2: Zero-Weight Stiffness Undefined

**What goes wrong:** `compute_stiffness_field` returns NaN for zero weights (log2(0) undefined)
**Why it happens:** Mathematical singularity at w=0
**How to avoid:** Replace NaN with 1.0 (neutral multiplier) or use minimum representable stiffness
**Warning signs:** NaN propagating through optimizer state

**Prevention strategy:**
```python
stiffness = torch.where(
    torch.isnan(stiffness),
    torch.ones_like(stiffness),
    stiffness
)
```

### Pitfall 3: Momentum Interaction with Preconditioning

**What goes wrong:** Stiffness-preconditioned gradients interact poorly with momentum
**Why it happens:** Momentum accumulates preconditioned gradients; stiffness changes as weights move
**How to avoid:** Two options: (a) precondition before momentum, or (b) precondition after momentum but before update
**Warning signs:** Training dynamics differ unexpectedly from theory

**Prevention strategy:**
Apply preconditioning consistently - recommended approach is before momentum:
```python
# Precondition gradient BEFORE moment update
precond_grad = grad * stiffness
exp_avg.mul_(beta1).add_(precond_grad, alpha=1 - beta1)
```

### Pitfall 4: E0M7 Format Special Case

**What goes wrong:** E0M7 has constant stiffness (1/128) but code may compute variable
**Why it happens:** E0M7 is fixed-point with uniform grid, not floating-point
**How to avoid:** The existing `compute_stiffness_field` already handles M=7 case
**Warning signs:** E0M7 behaves unexpectedly differently from floating-point formats

**Prevention strategy:**
The existing code already handles this:
```python
# In compute_stiffness_field (already implemented):
if mantissa_bits == 7:
    return torch.full_like(weights, 1.0 / 128.0)  # Constant stiffness
```

### Pitfall 5: Checkpoint Compatibility with Toggle

**What goes wrong:** Loading checkpoint with manifold_aware=True into optimizer with manifold_aware=False (or vice versa)
**Why it happens:** bit_position state exists in one but not other
**How to avoid:** State dict loading handles missing keys gracefully; extra keys are ignored
**Warning signs:** Missing keys warnings on load

**Prevention strategy:**
Initialize bit_position lazily in all modes (just set to zeros if not using):
```python
if "bit_position" not in state:
    state["bit_position"] = torch.zeros_like(p)
```

### Pitfall 6: Learning Rate Interaction

**What goes wrong:** Manifold-aware mode may need different learning rate than standard
**Why it happens:** Effective update size changes with stiffness preconditioning
**How to avoid:** Document that LR may need tuning; provide recommended starting points
**Warning signs:** Training unstable only in manifold mode, or vice versa

**Prevention strategy:**
Start with same LR and tune empirically. Consider stiffness normalization:
```python
# Optional: normalize stiffness to maintain similar effective LR
median_stiffness = stiffness.median()
stiffness = stiffness / median_stiffness
```

## Code Examples

Verified patterns from official sources and existing codebase:

### Complete ManifoldAdamW Skeleton
```python
# Source: PyTorch Adam implementation pattern + project requirements
import torch
from torch.optim import Optimizer
from altgrad.quantization.advanced_diagnostics import compute_stiffness_field

class ManifoldAdamW(Optimizer):
    """AdamW optimizer with optional stiffness-preconditioned updates.

    When manifold_aware=True, multiplies gradients by the local stiffness
    factor before computing Adam moments. This causes updates to move
    weights by a consistent number of ULPs rather than fixed real values.

    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (default: 1e-3)
        betas: Adam beta coefficients (default: (0.9, 0.999))
        eps: Numerical stability epsilon (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0.01)
        manifold_aware: Enable stiffness preconditioning (default: True)
        mantissa_bits: Format mantissa bits M (default: 2 for E5M2)
        max_stiffness: Maximum stiffness clamp (default: 1e6)

    Example:
        >>> optimizer = ManifoldAdamW(
        ...     model.parameters(),
        ...     lr=3e-4,
        ...     manifold_aware=True,
        ...     mantissa_bits=2,  # E5M2
        ... )
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        manifold_aware: bool = True,
        mantissa_bits: int = 2,
        max_stiffness: float = 1e6,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            manifold_aware=manifold_aware,
            mantissa_bits=mantissa_bits,
            max_stiffness=max_stiffness,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ManifoldAdamW does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["bit_position"] = torch.zeros_like(p)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Stiffness preconditioning (MANI-02)
                if group["manifold_aware"]:
                    stiffness = compute_stiffness_field(p.data, group["mantissa_bits"])
                    # Handle NaN (zero weights)
                    stiffness = torch.where(
                        torch.isnan(stiffness),
                        torch.ones_like(stiffness),
                        stiffness
                    )
                    # Clamp to prevent explosion
                    stiffness = stiffness.clamp(max=group["max_stiffness"])
                    # Precondition gradient
                    grad = grad * stiffness

                # Store weight before update (for bit-position tracking)
                weight_before = p.data.clone()

                # Decoupled weight decay (AdamW style)
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Update bit-position tracking (MANI-04)
                self._update_bit_position(state, weight_before, p.data)

        return loss

    def _update_bit_position(self, state: dict, before: torch.Tensor, after: torch.Tensor):
        """Track cumulative ULP movement."""
        inf_tensor = torch.full_like(before, float("inf"))
        ulp = torch.abs(torch.nextafter(before, inf_tensor) - before)
        safe_ulp = ulp.clamp(min=1e-45)
        delta_ulps = (after - before) / safe_ulp
        state["bit_position"] += delta_ulps
```

### Integration with Trainer
```python
# Source: Existing trainer.py pattern
# In trainer.py, modify configure_optimizers call:

def configure_optimizers(
    self,
    weight_decay: float,
    learning_rate: float,
    betas: tuple,
    device_type: str,
    manifold_aware: bool = False,
    mantissa_bits: int = 2,
):
    """Configure optimizer with optional manifold-aware mode."""
    # ... existing param group setup ...

    if manifold_aware:
        from altgrad.training.optimizer import ManifoldAdamW
        optimizer = ManifoldAdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            manifold_aware=True,
            mantissa_bits=mantissa_bits,
        )
    else:
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )

    return optimizer
```

### TrainConfig Extension
```python
# Source: Existing config.py pattern
# Add to TrainConfig dataclass:

# Manifold-aware optimizer (Phase 5)
use_manifold_aware: bool = False
manifold_mantissa_bits: int = 2  # M for S = 2^(floor(log2|w|) - M)
manifold_max_stiffness: float = 1e6  # Clamp to prevent explosion
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Uniform gradient steps | Per-weight preconditioning | Research ongoing | Addresses quantization grid non-uniformity |
| Post-quantize adjustment | Pre-quantize preconditioning | - | Proactive rather than reactive |
| Fixed-point bit tracking | Floating-point ULP tracking | - | More flexible for mixed formats |

**Current practice in FP8 training:**
- Most FP8 training keeps optimizer state in higher precision (BF16/FP32)
- Gradient scaling handles dynamic range, not grid non-uniformity
- Stiffness preconditioning is novel approach for this project

**Deprecated/outdated:**
- None identified - this is a novel research approach

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal stiffness clamp value**
   - What we know: Need some clamp to prevent gradient explosion
   - What's unclear: 1e6 is arbitrary; may need format-specific tuning
   - Recommendation: Make configurable, start with 1e6, tune empirically

2. **Preconditioning before vs after momentum**
   - What we know: Both mathematically valid, may have different dynamics
   - What's unclear: Which produces better training stability
   - Recommendation: Implement before momentum (cleaner), document for future ablation

3. **Learning rate adjustment for manifold mode**
   - What we know: Effective LR changes with stiffness preconditioning
   - What's unclear: Optimal LR for manifold vs standard mode
   - Recommendation: Start with same LR, document expected differences

4. **E7M0 behavior**
   - What we know: M=0 means S = 2^floor(log2|w|), purely exponential
   - What's unclear: Whether manifold-aware helps or hurts this extreme case
   - Recommendation: Test as expected failure/edge case, document results

5. **Bit-position accumulation semantics**
   - What we know: Track cumulative ULP movement for analysis
   - What's unclear: Whether to track absolute or signed ULP movement
   - Recommendation: Track signed (actual direction), can compute absolute in analysis

## Sources

### Primary (HIGH confidence)

**PyTorch Optimizer:**
- [torch.optim documentation](https://docs.pytorch.org/docs/stable/optim.html) - Base class interface
- [GitHub: torch/optim/optimizer.py](https://github.com/pytorch/pytorch/blob/main/torch/optim/optimizer.py) - Implementation reference
- [torch.optim.Optimizer.state_dict](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html) - Checkpoint serialization

**Existing Codebase:**
- `altgrad/quantization/advanced_diagnostics.py` - `compute_stiffness_field()` implementation
- `altgrad/quantization/diagnostics.py` - `BitStallDetector` implementation
- `altgrad/training/trainer.py` - Current optimizer integration pattern
- `altgrad/training/model.py` - `configure_optimizers()` pattern

### Secondary (MEDIUM confidence)

**Preconditioned Optimizers:**
- [PSGD (psgd_torch)](https://github.com/lixilinx/psgd_torch) - Preconditioned SGD patterns
- [ASDL Paper (arXiv:2305.04684)](https://arxiv.org/pdf/2305.04684) - Gradient preconditioning interface
- [torch_optimizer.shampoo](https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/shampoo.html) - Per-parameter state management

**Quantization-Aware Training:**
- [PyTorch QAT Blog](https://pytorch.org/blog/quantization-in-practice/) - STE gradient handling
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) - FP8 gradient considerations

### Tertiary (LOW confidence)

- [FP8-LM Paper](https://arxiv.org/html/2310.18313v2) - FP8 training framework (different focus)
- Various blog posts on custom optimizers (implementation details may vary)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using standard PyTorch Optimizer base class
- Architecture patterns: HIGH - Patterns directly from PyTorch and existing codebase
- Pitfalls: MEDIUM - Some based on theory, need empirical validation
- Code examples: HIGH - Based on PyTorch conventions and existing codebase patterns

**Research date:** 2026-01-22

**Valid until:** 60 days - optimizer APIs are stable, novel aspects need empirical validation

**Key takeaway for planner:**
1. Create `ManifoldAdamW` optimizer extending `torch.optim.Optimizer`
2. Reuse `compute_stiffness_field()` from Phase 4 for MANI-01
3. Implement stiffness preconditioning in step() for MANI-02
4. Add `manifold_aware` toggle for MANI-03
5. Track `bit_position` in per-parameter state for MANI-04
6. Update `TrainConfig` and `Trainer` to support manifold optimizer
7. Test toggle produces measurably different dynamics
