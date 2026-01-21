# Phase 01: Quantization Engine - Research

**Researched:** 2026-01-21
**Domain:** Custom FP8 quantization with PyTorch autograd integration
**Confidence:** HIGH

## Summary

The Quantization Engine phase requires implementing a standalone module for custom FP8 format quantization with gradient flow via Straight-Through Estimator (STE). This is NOT about using PyTorch's native FP8 types (torch.float8_e4m3fn, torch.float8_e5m2) but implementing custom exotic formats (E0M7, E1M6, E3M4, E5M2, E7M0) with full bit-level control.

The research reveals that the standard pattern involves: (1) Custom autograd.Function for STE gradient override, (2) Vectorized bit manipulation avoiding Python loops, (3) nn.Module wrappers managing quantization state via buffers not parameters, (4) Per-tensor scaling with exponential moving average (EMA) for amax tracking, and (5) Comprehensive testing with gradcheck and round-trip validation.

The existing codebase has already implemented the format registry (formats.py) and quantize/dequantize operations (ops.py). The remaining work focuses on per-tensor scaling with delayed amax tracking and bit-stall detection for gradient stability analysis.

**Primary recommendation:** Use nn.Module with register_buffer for scale factors and amax history, implement EMA-based delayed scaling following NVIDIA's FP8 training patterns, and test gradient flow with torch.autograd.gradcheck.

## Standard Stack

The established libraries/tools for custom FP8 quantization in PyTorch:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9+ | Deep learning framework with autograd | Native autograd.Function for custom gradients, established quantization infrastructure |
| pytest | 7.0+ | Testing framework | Standard for PyTorch testing, supports parametrization critical for multi-format testing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | Latest | Numerical operations reference | Validating bit manipulation logic, generating test cases |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom autograd | JAX custom_vjp | JAX has cleaner gradient API but PyTorch ecosystem is standard for this domain |
| PyTorch native FP8 | torch.float8_e4m3fn | Only supports standard IEEE formats, cannot implement E0M7/E1M6/E3M4/E7M0 |
| NVIDIA pytorch-quantization | Custom implementation | NVIDIA toolkit is INT8-focused and tied to TensorRT, doesn't support custom FP8 formats |

**Installation:**
```bash
# Already in pyproject.toml - no additional dependencies needed
pip install -e ".[dev]"
```

## Architecture Patterns

### Recommended Project Structure
```
altgrad/quantization/
├── formats.py           # FP8Format registry with transfer functions ✓ IMPLEMENTED
├── ops.py              # Quantize/dequantize autograd functions ✓ IMPLEMENTED
├── modules.py          # nn.Module wrappers for stateful quantization (TODO)
└── calibration.py      # Amax tracking and scale computation (TODO)

tests/
├── test_formats.py     # Format correctness tests ✓ IMPLEMENTED
├── test_ops.py        # STE gradient tests ✓ IMPLEMENTED
├── test_modules.py    # Module state management tests (TODO)
└── test_calibration.py # Amax tracking and scaling tests (TODO)
```

### Pattern 1: Custom Autograd Function for STE

**What:** Implement quantization as torch.autograd.Function with identity backward pass
**When to use:** Any non-differentiable operation needing gradient flow (quantization, rounding, thresholding)

**Example:**
```python
# Source: https://docs.pytorch.org/docs/stable/notes/extending.html
class QuantizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, format: FP8Format, scale: Tensor) -> Tensor:
        # ctx.save_for_backward() not needed for STE - no saved tensors required
        # Non-tensor arguments stored directly: ctx.format = format
        x_scaled = x / scale
        x_quantized = _vectorized_quantize(x_scaled, format)
        return x_quantized * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        # STE: gradient passes through unchanged
        return grad_output, None, None  # None for format and scale
```

**Key insight:** STE doesn't need ctx.save_for_backward() because backward is independent of forward values. This is already correctly implemented in ops.py.

### Pattern 2: State Management with Buffers

**What:** Use register_buffer for scale factors and amax history (not parameters)
**When to use:** Quantization metadata that must persist in state_dict but isn't learnable

**Example:**
```python
# Source: https://docs.pytorch.org/docs/stable/notes/modules.html
class PerTensorQuantizer(nn.Module):
    def __init__(self, format: FP8Format, amax_history: int = 1024):
        super().__init__()
        self.format = format  # Store format spec directly on module

        # Persistent buffers - saved in state_dict
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('amax_history', torch.zeros(amax_history))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))

        # Non-persistent buffers - not saved (optional for ephemeral state)
        self.register_buffer('current_amax', torch.tensor(0.0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # Update amax history
        current_amax = x.abs().max()
        idx = self.history_ptr % len(self.amax_history)
        self.amax_history[idx] = current_amax
        self.history_ptr += 1

        # Compute scale from delayed history
        self.scale = self._compute_scale()

        return quantize(x, self.format, self.scale)
```

**Critical distinction:** Buffers are serialized but not optimized. Parameters receive gradients. Scale factors are neither learnable nor transient, making buffers the correct choice.

### Pattern 3: EMA-Based Delayed Scaling

**What:** Compute scale factors from exponential moving average of historical amax values
**When to use:** Dynamic range tracking across training iterations with robustness to outliers

**Example:**
```python
# Source: https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/
class DelayedAmaxTracker:
    def __init__(self, averaging_constant: float = 0.01, history_len: int = 1024):
        self.averaging_constant = averaging_constant  # For EMA computation
        self.history_len = history_len
        self.amax_history = torch.zeros(history_len)
        self.ptr = 0

    def update(self, tensor: Tensor) -> None:
        """Update history with current amax value."""
        current_amax = tensor.abs().max()
        self.amax_history[self.ptr % self.history_len] = current_amax
        self.ptr += 1

    def compute_scale(self, format: FP8Format) -> float:
        """Compute scale from historical amax using EMA smoothing."""
        # Delayed scaling: use historical data, not current batch
        valid_history = self.amax_history[:min(self.ptr, self.history_len)]

        if len(valid_history) == 0:
            return 1.0

        # EMA over history for robustness to outliers
        ema = valid_history[0]
        for amax in valid_history[1:]:
            ema = self.averaging_constant * amax + (1 - self.averaging_constant) * ema

        # Scale to fit format's representable range
        max_representable = format.max_representable_value
        return ema / max_representable if ema > 0 else 1.0
```

**Tradeoff:** Delayed scaling is robust to outliers but assumes stable distributions. Current scaling (using immediate batch stats) adapts faster but is more sensitive to noise. Delayed is standard for FP8 training.

### Pattern 4: Vectorized Quantization

**What:** Process entire tensors without Python loops using PyTorch operations
**When to use:** Always - Python loops are 100-1000x slower for numerical operations

**Example:**
```python
# Already implemented correctly in ops.py
def _quantize_floating_point(x: Tensor, format: FP8Format) -> Tensor:
    # CORRECT: Uses torch operations on entire tensor
    sign = torch.sign(x_flat)
    abs_x = torch.abs(x_flat)

    # Process denormal and normal values with masks
    is_denorm = abs_x < min_normal
    quant_result = torch.zeros_like(abs_x)
    quant_result[is_denorm] = _quantize_denorm_vec(abs_x[is_denorm], format, ...)
    quant_result[~is_denorm] = _quantize_normal_vec(abs_x[~is_denorm], format, ...)

    # WRONG: for val in x: quantize(val)  # Never do this
```

**Performance note:** Existing ops.py implementation is correctly vectorized. Performance test shows <1s for 1M elements (test_large_tensor_performance).

### Pattern 5: Bit-Stall Detection

**What:** Track when round(b + delta_b) == b despite non-zero gradient
**When to use:** Debugging gradient flow, detecting when quantization resolution is too coarse

**Example:**
```python
class BitStallCounter:
    """Tracks when quantized parameter updates round to zero."""

    def __init__(self):
        self.stall_count = 0
        self.update_count = 0

    def check_update(self,
                     param_before: Tensor,
                     param_after: Tensor,
                     gradient: Tensor,
                     format: FP8Format,
                     scale: Tensor) -> Dict[str, int]:
        """
        Check if gradient-based updates stalled due to quantization.

        Returns dict with 'stalled' and 'total' counts.
        """
        # Quantize both before and after applying gradient update
        q_before = quantize(param_before, format, scale)
        q_after = quantize(param_after, format, scale)

        # Stall when quantized values don't change despite gradient
        stalled = (q_before == q_after) & (gradient.abs() > 1e-8)

        self.stall_count += stalled.sum().item()
        self.update_count += gradient.numel()

        return {
            'stalled': stalled.sum().item(),
            'total': gradient.numel(),
            'stall_rate': self.stall_count / max(self.update_count, 1)
        }
```

**Why this matters:** E7M0 (powers of 2 only) and E0M7 (fixed-point) have very coarse quantization. Bit-stall detection quantifies how often gradient updates are "lost" to rounding.

### Anti-Patterns to Avoid

- **In-place modification of gradients:** Never modify grad_output in backward - create new tensors
- **Saving tensors unnecessarily:** STE doesn't need ctx.save_for_backward() - only use when backward truly depends on forward values
- **Python loops over tensors:** Always vectorize with torch operations, never iterate elements
- **Parameters for scale factors:** Use buffers with register_buffer, not nn.Parameter
- **Ignoring denormalized numbers:** E3M4, E5M2 have denorms - handle explicitly or accept clamping to zero

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Moving average tracking | Manual history buffer with indexing | torch.ao.quantization.observer.MovingAverageMinMaxObserver pattern | Edge cases: warmup period, buffer wraparound, serialization already solved |
| Gradient correctness testing | Manual finite differences | torch.autograd.gradcheck | Handles epsilon selection, tolerance, higher-order derivatives, edge cases |
| Test parametrization | Custom test generation loops | @pytest.mark.parametrize | Built-in cartesian product, test IDs, xfail marking, better error reporting |
| Tensor device/dtype handling | Manual .to() calls | nn.Module.register_buffer | Automatically moves buffers with module.to(device), handles state_dict |
| Round-to-nearest-even | Manual rounding logic | Already implemented in formats.py._round_to_nearest_even | IEEE 754 tie-breaking is subtle - existing implementation is correct |

**Key insight:** The existing codebase (formats.py, ops.py) has already correctly implemented the hard parts: round-to-nearest-even rounding, denormalized number handling, and vectorized quantization. Don't rewrite these.

## Common Pitfalls

### Pitfall 1: Forgetting to Test Gradient Flow

**What goes wrong:** Custom autograd functions work in forward but silently break gradients
**Why it happens:** PyTorch doesn't validate custom backward implementations at definition time
**How to avoid:** Always test with torch.autograd.gradcheck immediately after implementing

**Warning signs:**
- Loss doesn't decrease during training
- Gradients are None or all zeros
- Model parameters don't update

**Prevention strategy:**
```python
# Source: https://docs.pytorch.org/docs/stable/notes/extending.html
def test_quantize_gradcheck():
    """Validate gradient correctness with finite differences."""
    x = torch.randn(10, dtype=torch.double, requires_grad=True)
    scale = torch.tensor(1.0, dtype=torch.double)

    # gradcheck uses finite differences to validate analytical gradient
    # Use double precision for numerical stability
    assert torch.autograd.gradcheck(
        lambda inp: quantize(inp, E5M2, scale),
        x,
        eps=1e-6,
        atol=1e-4
    )
```

**Critical:** Use torch.double (FP64) for gradcheck, not torch.float32. Finite differences need higher precision.

### Pitfall 2: Scale Factor Growing Unbounded

**What goes wrong:** Amax history accumulates outliers, scale factor explodes, all values quantize to zero
**Why it happens:** History buffer doesn't handle outliers or reset after distribution shift
**How to avoid:** Use EMA smoothing and cap maximum scale to prevent runaway growth

**Warning signs:**
- Scale factor > 1000 after a few iterations
- All quantized values near zero
- Training becomes unstable or diverges

**Prevention strategy:**
```python
def compute_scale_safe(self, format: FP8Format) -> float:
    """Compute scale with outlier protection."""
    valid_history = self.amax_history[:min(self.ptr, self.history_len)]

    if len(valid_history) == 0:
        return 1.0

    # Use percentile instead of max to ignore outliers
    robust_amax = torch.quantile(valid_history, 0.99)  # 99th percentile

    # EMA smoothing
    if hasattr(self, 'scale_ema'):
        self.scale_ema = 0.01 * robust_amax + 0.99 * self.scale_ema
    else:
        self.scale_ema = robust_amax

    # Cap maximum scale to prevent explosion
    max_scale = 1000.0
    scale = min(self.scale_ema / format.max_representable_value, max_scale)

    return max(scale, 1e-8)  # Minimum scale to prevent division by zero
```

### Pitfall 3: Not Handling Zero Gradients in backward

**What goes wrong:** backward returns None for some inputs, breaks computational graph
**Why it happens:** Forgetting to return None for non-differentiable arguments (format, scale)
**How to avoid:** Return exactly one gradient per forward argument, None for non-tensors

**Warning signs:**
- RuntimeError: "grad can be implicitly created only for scalar outputs"
- TypeError: "expected Tensor as element X in argument Y, but got NoneType"

**Prevention strategy:**
```python
# CORRECT - returns 3 values for 3 forward arguments
@staticmethod
def backward(ctx, grad_output):
    return grad_output, None, None  # For x, format, scale

# WRONG - missing return values
@staticmethod
def backward(ctx, grad_output):
    return grad_output  # Missing None for format and scale - breaks graph

# WRONG - returning gradient for non-tensor
@staticmethod
def backward(ctx, grad_output):
    return grad_output, torch.zeros_like(format), None  # format is not a tensor
```

**Rule:** backward must return exactly len(forward_args) values. Non-tensor/non-differentiable arguments get None.

### Pitfall 4: Testing Only Positive Values

**What goes wrong:** Quantization breaks for negative values or signed zero
**Why it happens:** Sign bit handling is separate from magnitude quantization - easy to get wrong
**How to avoid:** Explicitly test negative values, -0.0, and sign preservation

**Warning signs:**
- Negative values quantize incorrectly
- Sign bit flipped for small negative values
- -0.0 becomes +0.0 (may or may not matter)

**Prevention strategy:**
```python
@pytest.mark.parametrize("value", [
    -1.0, -0.5, -0.0, 0.0, 0.5, 1.0,  # Sign boundary cases
    -1e-10, 1e-10,  # Near-zero with sign
    -float('inf'), float('inf'),  # Infinity with sign
])
def test_quantize_sign_handling(value):
    """Test sign preservation across quantization."""
    x = torch.tensor([value])
    y = quantize(x, E5M2, torch.tensor(1.0))

    # Check sign is preserved (allowing for clamping)
    if value != 0:
        assert torch.sign(y) == torch.sign(x)
```

### Pitfall 5: Denormalized Numbers Break Round-Trip

**What goes wrong:** Very small values don't round-trip correctly (to_bits(to_real(bits)) != bits)
**Why it happens:** Denormalized encoding differs from normalized - different exponent interpretation
**How to avoid:** Explicit denorm handling in both to_bits and to_real with correct bias computation

**Warning signs:**
- Round-trip tests fail for small bit indices (0-10)
- Smallest representable values incorrect
- Gradual underflow doesn't work (cliff to zero)

**Prevention strategy:**
```python
def test_denorm_roundtrip():
    """Test round-trip for denormalized values."""
    # E5M2: denorms are bits 0b00000001 to 0b00000011 (exp=0, mantissa>0)
    for bits in range(1, 4):  # Skip zero
        value = E5M2.to_real(bits)
        recovered_bits = E5M2.to_bits(value)
        assert bits == recovered_bits, f"Denorm {bits} failed round-trip"
```

**Already solved:** formats.py correctly handles denorms in to_real (lines 104-107) and to_bits (lines 236-251). Don't reimplement.

### Pitfall 6: Buffer Not Updating During forward

**What goes wrong:** Amax history stays empty, scale never adapts, quantization uses wrong range
**Why it happens:** Forgetting to call update methods during forward pass, treating it as inference-only
**How to avoid:** Update buffers during forward in training mode, use training flag to disable in eval

**Warning signs:**
- Scale factor is always 1.0
- Amax history buffer is all zeros
- Quantization range doesn't adapt to data

**Prevention strategy:**
```python
class PerTensorQuantizer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # CORRECT: Update history during training
        if self.training:
            current_amax = x.abs().max()
            idx = self.history_ptr % len(self.amax_history)
            self.amax_history[idx] = current_amax
            self.history_ptr += 1
            self.scale = self._compute_scale()

        # Use current scale (updated if training, frozen if eval)
        return quantize(x, self.format, self.scale)
```

**Note:** History update must happen in forward, not in a separate calibration phase, because it needs to track live training statistics.

## Code Examples

Verified patterns from official sources:

### Testing Round-Trip Correctness
```python
# Source: Existing test_formats.py (lines 282-296)
@pytest.mark.parametrize("format_name", ["E0M7", "E1M6", "E3M4", "E5M2", "E7M0"])
def test_roundtrip_all_bit_patterns(format_name):
    """For every bit pattern, to_bits(to_real(bits)) == bits."""
    fmt = FORMAT_REGISTRY[format_name]
    for bits in range(256):
        value = fmt.to_real(bits)
        # Skip special values (NaN, Inf) and negative zero
        if math.isnan(value) or math.isinf(value):
            continue
        if value == 0.0 and bits != 0:
            continue
        recovered_bits = fmt.to_bits(value)
        assert recovered_bits == bits
```

### Testing STE Gradient Flow
```python
# Source: Existing test_ops.py (lines 70-84)
def test_ste_gradient_passthrough():
    """STE passes gradient unchanged through quantization."""
    x = torch.randn(10, requires_grad=True)
    scale = torch.tensor(1.0)

    y = quantize(x, E5M2, scale)
    loss = y.sum()
    loss.backward()

    # Gradient should be all ones (d(sum)/dx = 1 for each element)
    expected_grad = torch.ones_like(x)
    assert x.grad is not None
    assert torch.allclose(x.grad, expected_grad)
```

### Computing Scale from Format Range
```python
# Source: formats.py (lines 46-63) + NVIDIA FP8 blog
def compute_scale_for_format(tensor: Tensor, format: FP8Format) -> float:
    """
    Compute scale factor to fit tensor's dynamic range into format.

    Scale factor s maps tensor values to format range:
        quantized_value = quantize(tensor_value / s, format) * s
    """
    # Get tensor's dynamic range
    amax = tensor.abs().max().item()

    # Get format's representable range
    max_representable = format.max_representable_value

    # Scale so amax maps to ~75% of max representable (headroom for spikes)
    if amax == 0:
        return 1.0

    scale = amax / (0.75 * max_representable)
    return scale
```

### Handling Denormalized Numbers
```python
# Source: formats.py (lines 104-107, 236-251)
# This is already correctly implemented - reference for understanding

# In to_real:
if exponent == 0 and mantissa != 0:
    # Denorm: value = 2^(1-bias) * (mantissa / 2^M)
    value = (2 ** (1 - self.bias)) * (mantissa / (1 << self.mantissa_bits))
    return -value if sign else value

# In to_bits (denorm path):
def _to_bits_denorm(self, sign: int, abs_value: float) -> int:
    """Convert to bits for denormalized numbers."""
    # mantissa = value * 2^(M + bias - 1)
    scale = 2 ** (self.mantissa_bits + self.bias - 1)
    mantissa_float = abs_value * scale

    mantissa = self._round_to_nearest_even(mantissa_float)
    max_mantissa = (1 << self.mantissa_bits) - 1

    # Handle overflow to smallest normal
    if mantissa > max_mantissa:
        return (sign << 7) | (1 << self.mantissa_bits)

    return (sign << 7) | mantissa
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| INT8 quantization | FP8 quantization | 2022-2024 (H100 release) | FP8 preserves dynamic range better than INT8, 92.64% pass rate vs 65.87% for INT8 |
| Static amax (max value) | EMA smoothing of amax | 2023-2025 | Robustness to outliers, more stable training |
| Per-channel scaling | Per-tensor + per-block | 2024-2025 (MXFP8) | Simpler implementation for per-tensor, per-block for extreme cases |
| Current scaling (immediate) | Delayed scaling (historical) | 2023-2024 | Delayed is more stable for FP8 training, avoids outlier spikes |
| torch.quantization | torch.ao.quantization (torchao) | 2025-2026 | API consolidation, torch.ao.quantization deprecated in PyTorch 2.10 |

**Deprecated/outdated:**
- torch.quantization: Moved to torch.ao.quantization, planned deletion in PyTorch 2.10
- NVIDIA pytorch-quantization (TensorRT 8.x toolkit): Focused on INT8, doesn't support custom FP8 formats
- Current scaling for FP8: Delayed scaling is now standard practice (NVIDIA's recommendation)

**Bleeding edge (2026):**
- NVFP4 (E2M1): 4-bit format on Blackwell GPUs, even more aggressive quantization
- torch.compile() support for quantization: Better performance through compiler fusion
- Per-block scaling: MXFP8 uses 32-element blocks with E8M0 scale factors (power-of-2 only)

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal amax history buffer size**
   - What we know: NVIDIA toolkit defaults vary (256-1024), moving average uses averaging_constant=0.01
   - What's unclear: Optimal size depends on training dynamics and format coarseness
   - Recommendation: Start with 1024, make configurable, track scale stability to tune

2. **Bit-stall threshold for failure**
   - What we know: Bit-stall happens when round(b + delta_b) == b, measurable as stall rate
   - What's unclear: What stall rate indicates unrecoverable training degradation (10%? 50%? 90%?)
   - Recommendation: Implement detection as proposed, empirically determine threshold through experiments

3. **E7M0 viability threshold**
   - What we know: E7M0 (powers of 2 only) is deliberately experimental, expected to fail
   - What's unclear: At what layer width/depth does E7M0 become completely non-viable
   - Recommendation: This is a research question, not an implementation question - detection is the goal

4. **Double backward support**
   - What we know: STE typically doesn't support double backward (gradient of gradient)
   - What's unclear: Whether this research needs second-order derivatives for analysis
   - Recommendation: Mark backward with @once_differentiable unless second-order is needed

5. **Persistent vs non-persistent buffers for amax history**
   - What we know: Persistent buffers save to state_dict, non-persistent don't
   - What's unclear: Should amax history persist across save/load or reset fresh
   - Recommendation: Make history persistent for reproducibility, add reset() method for fresh calibration

## Sources

### Primary (HIGH confidence)

Official PyTorch Documentation:
- [Extending PyTorch - Custom autograd Functions](https://docs.pytorch.org/docs/stable/notes/extending.html)
- [nn.Module Notes - Buffers and Parameters](https://docs.pytorch.org/docs/stable/notes/modules.html)
- [Gradcheck Mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)
- [Double Backward with Custom Functions Tutorial](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)
- [MovingAverageMinMaxObserver API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html)
- [torch.ao.quantization Overview](https://docs.pytorch.org/ao/stable/quantization_overview.html)

NVIDIA Technical Documentation:
- [Per-Tensor and Per-Block Scaling Strategies for FP8 Training](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [pytorch-quantization toolkit docs](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-803/pytorch-quantization-toolkit/docs/index.html)
- [Transformer Engine FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

### Secondary (MEDIUM confidence)

Academic and Technical Articles:
- [FP8 Quantization: The Power of the Exponent (arXiv:2208.09225)](https://arxiv.org/abs/2208.09225)
- [Practical Quantization in PyTorch (PyTorch Blog)](https://pytorch.org/blog/quantization-in-practice/)
- [Intuitive Explanation of Straight-Through Estimators](https://hassanaskary.medium.com/intuitive-explanation-of-straight-through-estimators-with-pytorch-implementation-71d99d25d9d0)
- [Working with pytest on PyTorch (Quansight Labs)](https://labs.quansight.org/blog/2021/06/pytest-pytorch)

### Tertiary (LOW confidence - marked for validation)

Community Resources:
- PyTorch Forums discussions on STE implementation patterns
- GitHub issues on quantization errors and edge cases
- Medium articles on FP8 implementation (not verified against official docs)

**Note on WebSearch findings:** Most WebSearch results were cross-verified against official PyTorch and NVIDIA documentation. Unverified WebSearch-only findings are marked with LOW confidence.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch and pytest are definitively standard for this domain
- Architecture patterns: HIGH - Verified against official PyTorch docs and existing correct implementation
- Pitfalls: HIGH - Based on official documentation warnings and common error patterns
- Amax tracking implementation: MEDIUM - NVIDIA docs describe approach but optimal parameters are empirical
- Bit-stall thresholds: LOW - This is a research question without established answers

**Research date:** 2026-01-21

**Valid until:** 30 days for PyTorch API stability, 7 days for FP8 quantization best practices (fast-moving field)

**Implementation status:**
- Format registry: ✓ COMPLETE (formats.py)
- Quantize/dequantize ops: ✓ COMPLETE (ops.py)
- STE gradient flow: ✓ COMPLETE (ops.py)
- Per-tensor scaling module: ⧗ TODO (modules.py)
- Amax history tracking: ⧗ TODO (calibration.py)
- Bit-stall detection: ⧗ TODO (new module or extension)

**Key takeaway for planner:** The hard parts (bit manipulation, STE, vectorization) are done. Focus remaining work on state management (nn.Module wrappers), calibration (amax tracking), and analysis tools (bit-stall detection).
