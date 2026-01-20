# Architecture Research: FP8 Training Test Bench

**Domain:** Quantized transformer training research platform
**Researched:** 2026-01-20
**Confidence:** MEDIUM to HIGH

## Executive Summary

FP8 training test benches follow a **simulated quantization architecture** where tensors are stored in high precision (FP32/BF16) but quantized at specific boundaries to emulate low-precision behavior. The architecture consists of four major subsystems: (1) **Quantization Engine** with format-specific encode/decode, (2) **Training Loop Integration** with strategic fake-quantization insertion points, (3) **State Tracking System** for monitoring bit-level statistics, and (4) **nanoGPT Backbone** with minimal surgical modifications.

**Key finding:** Modern QAT approaches use **fake quantization with Straight-Through Estimators (STE)** rather than true bit-width casting, enabling gradient flow while simulating numerical effects. This is implemented via PyTorch hooks or custom autograd functions at layer boundaries.

**Critical insight for roadmap:** Build quantization engine FIRST (independent component), then integrate with nanoGPT via minimal wrapper layers. This enables parallel development and easy format experimentation.

---

## Component Overview

### 1. Quantization Engine (Core Component)

**Responsibility:** Simulate arbitrary FP8 formats (E1M6, E3M4, E5M2, E7M0) with configurable quantize/dequantize operations.

**Sub-components:**
- **Format Registry:** Defines mantissa/exponent splits, dynamic range, special values (NaN/Inf handling)
- **Scaling Factor Calculator:** Per-tensor or per-channel scaling using max/amax statistics
- **Quantize Function:** FP32 → FP8 simulation (round to nearest representable value)
- **Dequantize Function:** FP8 simulation → FP32 (restore to compute dtype)
- **STE Gradient Override:** Custom autograd function that passes gradients unchanged

**Interface:**
```python
class FP8Format:
    def __init__(self, exponent_bits: int, mantissa_bits: int)
    def quantize(self, tensor: torch.Tensor, scale: float) -> torch.Tensor
    def dequantize(self, tensor: torch.Tensor, scale: float) -> torch.Tensor
    def compute_scale(self, tensor: torch.Tensor, method: str) -> float
```

**Why independent:** Can be unit-tested separately, reused across different model architectures, and easily extended to new formats.

---

### 2. Fake Quantization Wrapper Layers

**Responsibility:** Inject quantization/dequantization at strategic layer boundaries without modifying core nanoGPT code.

**Implementation pattern (from PyTorch AO/TorchAO):**
```python
class QuantizedLinear(nn.Module):
    def __init__(self, original_linear, weight_format, act_format):
        self.linear = original_linear
        self.weight_format = weight_format
        self.act_format = act_format
        self.weight_scale = None
        self.act_scale = None

    def forward(self, x):
        # Quantize activations
        x_scale = self.act_format.compute_scale(x, method='dynamic')
        x_quant = self.act_format.quantize(x, x_scale)
        x_dequant = self.act_format.dequantize(x_quant, x_scale)

        # Quantize weights (can cache scale for static quant)
        if self.weight_scale is None:
            self.weight_scale = self.weight_format.compute_scale(
                self.linear.weight, method='static'
            )
        w_quant = self.weight_format.quantize(self.linear.weight, self.weight_scale)
        w_dequant = self.weight_format.dequantize(w_quant, self.weight_scale)

        # Compute with dequantized (FP32) values
        return F.linear(x_dequant, w_dequant, self.linear.bias)
```

**Gradient flow:** The `quantize/dequantize` operations use STE, so `∂loss/∂x_dequant ≈ ∂loss/∂x` (identity passthrough).

---

### 3. Training Loop Integration Points

Based on research into NVIDIA Transformer Engine and PyTorch QAT patterns, quantization occurs at **6 strategic boundaries**:

#### Forward Pass Quantization Points

| Location | What to Quantize | Format Choice | Rationale |
|----------|------------------|---------------|-----------|
| **1. Attention QKV projections** | Weight + Activations | E4M3 (higher precision) | Critical for model accuracy; outliers in attention scores |
| **2. Attention output projection** | Weight + Activations | E4M3 | Preserves gradient signal for residual connections |
| **3. MLP FC1 (expansion)** | Weight + Activations | E4M3 or E3M4 | Large matrices, high arithmetic intensity |
| **4. MLP FC2 (contraction)** | Weight + Activations | E4M3 or E3M4 | Less sensitive than attention layers |
| **5. Classifier head** | Weight only | Mixed (experiment) | Final layer; test lower precision viability |

#### Backward Pass Quantization Points

| Location | What to Quantize | Format Choice | Rationale |
|----------|------------------|---------------|-----------|
| **6. Gradient tensors** | ∂loss/∂weights, ∂loss/∂activations | E5M2 (higher dynamic range) | Gradients span wider range, need exponent bits |

**Per-layer mixed precision:** The test bench should support assigning DIFFERENT FP8 formats to different layers. For example:
- Attention layers: E4M3 (3 mantissa bits for precision)
- MLP layers: E3M4 (4 mantissa bits, more precision)
- Gradients: E5M2 (5 exponent bits, higher dynamic range)

This is the **core research question**: Which format works best WHERE?

---

### 4. NanoGPT Integration Strategy

#### Minimal Modification Approach

**Do NOT fork nanoGPT.** Instead, use **model surgery** via wrapper functions:

```python
# In your test bench initialization code
from nanogpt.model import GPT

def quantize_model(model, layer_format_map):
    """Replace Linear layers with QuantizedLinear wrappers."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Determine format based on layer type
            if 'attn' in name:
                format_config = layer_format_map['attention']
            elif 'mlp' in name:
                format_config = layer_format_map['mlp']
            elif 'lm_head' in name:
                format_config = layer_format_map['classifier']
            else:
                continue  # Don't quantize embeddings

            # Replace in-place
            parent = _get_parent_module(model, name)
            setattr(parent, name.split('.')[-1],
                    QuantizedLinear(module, **format_config))
    return model

# Usage
model = GPT.from_pretrained('gpt2')
model = quantize_model(model, layer_format_map={
    'attention': {'weight_format': FP8Format(4, 3), 'act_format': FP8Format(4, 3)},
    'mlp': {'weight_format': FP8Format(3, 4), 'act_format': FP8Format(3, 4)},
    'classifier': {'weight_format': FP8Format(5, 2), 'act_format': None}
})
```

**Why this works:**
- nanoGPT's `model.py` is unchanged
- Quantization logic is self-contained
- Easy to disable (just don't call `quantize_model()`)
- Supports per-layer format experimentation

---

### 5. Custom Optimizer Integration (Stiffness Factor)

The project requires a **custom optimizer with stiffness factor injection**. Based on the Muon optimizer pattern from modded-nanogpt, here's the integration strategy:

#### Option A: Extend AdamW (Simpler)

```python
class StiffnessAdamW(torch.optim.AdamW):
    def __init__(self, params, lr, stiffness_factor=1.0, **kwargs):
        super().__init__(params, lr, **kwargs)
        self.stiffness_factor = stiffness_factor

    def step(self, closure=None):
        # Inject stiffness before parameter update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Scale gradient by stiffness factor
                p.grad = p.grad * self.stiffness_factor

        # Call parent optimizer step
        return super().step(closure)
```

**Usage in nanoGPT's train.py:**
```python
# Replace optimizer creation
optimizer = StiffnessAdamW(
    model.parameters(),
    lr=learning_rate,
    stiffness_factor=config.stiffness_factor,
    betas=(0.9, 0.95),
    weight_decay=0.1
)
```

#### Option B: Wrap Existing Optimizer (More Flexible)

```python
class StiffnessWrapper:
    def __init__(self, base_optimizer, stiffness_fn):
        self.optimizer = base_optimizer
        self.stiffness_fn = stiffness_fn  # Function: iter_num -> float

    def step(self, iter_num):
        stiffness = self.stiffness_fn(iter_num)
        # Modify gradients before step
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad * stiffness
        self.optimizer.step()
        self.optimizer.zero_grad()
```

**Why wrapper:** Allows dynamic stiffness schedules (e.g., decay over training), doesn't require subclassing each optimizer type.

---

### 6. Bit-Position Tracking System

**Responsibility:** Monitor which mantissa/exponent bits are "active" (non-zero) during training to understand format utilization.

**Architecture:**

```python
class BitTracker:
    def __init__(self, format_spec):
        self.format = format_spec
        self.bit_usage_histogram = defaultdict(int)  # bit_index -> activation_count

    def track_tensor(self, tensor_name, quantized_tensor):
        """Log bit-level statistics for a quantized tensor."""
        # Convert FP8-simulated values to binary representation
        binary_rep = self._to_binary(quantized_tensor)

        # Count active bits per position
        for bit_idx in range(8):  # FP8 has 8 bits
            active = (binary_rep & (1 << bit_idx)) != 0
            self.bit_usage_histogram[f"{tensor_name}_bit{bit_idx}"] += active.sum().item()

    def report(self):
        """Return bit utilization statistics."""
        return {
            'mantissa_utilization': self._compute_mantissa_usage(),
            'exponent_utilization': self._compute_exponent_usage(),
            'overflow_events': self.overflow_count
        }
```

**Integration with training loop:**
```python
# In train.py, after each forward pass
if iter_num % config.track_interval == 0:
    for name, module in model.named_modules():
        if hasattr(module, 'last_quantized_weight'):
            bit_tracker.track_tensor(name, module.last_quantized_weight)
```

**What to track:**
- **Bit utilization per layer:** Which mantissa/exponent bits are actually used?
- **Dynamic range coverage:** Are we hitting the format's limits (overflow/underflow)?
- **Gradient magnitude distribution:** Do gradients need more exponent bits (E5M2)?

**Logging:** Use WandB to log these statistics:
```python
wandb.log({
    'bit_tracking/layer_attention_qkv/mantissa_bit_2': usage_count,
    'bit_tracking/layer_attention_qkv/exponent_saturation': overflow_pct
})
```

---

## Data Flow Architecture

### Complete Forward/Backward Flow with Quantization

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Iteration                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. DATA LOADING                                                 │
│    - Fetch batch (X, Y) from EurLex dataset                    │
│    - Move to GPU                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FORWARD PASS (with FP8 simulation)                          │
│                                                                 │
│  X (BF16/FP32) ─────────────────────────────────────────────┐  │
│       │                                                      │  │
│       ▼                                                      │  │
│  ┌─────────────────────────────────┐                        │  │
│  │ Embedding Layer (no quant)      │                        │  │
│  └─────────────────────────────────┘                        │  │
│       │                                                      │  │
│       ▼                                                      │  │
│  ┌──────────────────────────────────────────────────────┐   │  │
│  │ FOR EACH TRANSFORMER LAYER:                          │   │  │
│  │                                                       │   │  │
│  │  ┌─────────────────────────────────────────────┐     │   │  │
│  │  │ Attention Block                             │     │   │  │
│  │  │  • Quantize input activations (E4M3)       │     │   │  │
│  │  │  • Quantize QKV weights (E4M3)             │     │   │  │
│  │  │  • Dequantize both to FP32                 │     │   │  │
│  │  │  • Compute attention with FP32             │     │   │  │
│  │  │  • [TRACK: bit usage, scale factors]       │     │   │  │
│  │  └─────────────────────────────────────────────┘     │   │  │
│  │       │                                               │   │  │
│  │       ▼                                               │   │  │
│  │  ┌─────────────────────────────────────────────┐     │   │  │
│  │  │ MLP Block                                   │     │   │  │
│  │  │  • Quantize input activations (E3M4)       │     │   │  │
│  │  │  • Quantize FC weights (E3M4)              │     │   │  │
│  │  │  • Dequantize both to FP32                 │     │   │  │
│  │  │  • Compute MLP with FP32                   │     │   │  │
│  │  │  • [TRACK: different format, compare]      │     │   │  │
│  │  └─────────────────────────────────────────────┘     │   │  │
│  │                                                       │   │  │
│  └──────────────────────────────────────────────────────┘   │  │
│       │                                                      │  │
│       ▼                                                      │  │
│  ┌─────────────────────────────────┐                        │  │
│  │ Classifier Head (mixed precision)│                       │  │
│  │  • Test E5M2 or E7M0 here       │                       │  │
│  └─────────────────────────────────┘                        │  │
│       │                                                      │  │
│       ▼                                                      │  │
│    logits (FP32) ───> LOSS                                  │  │
│                                                              │  │
└──────────────────────────────────────────────────────────────┘  │
                              │                                   │
                              ▼                                   │
┌─────────────────────────────────────────────────────────────────┐
│ 3. BACKWARD PASS (with gradient quantization)                  │
│                                                                 │
│  ∂loss/∂logits (FP32) ─────────────────────────────────────┐   │
│       │                                                     │   │
│       ▼                                                     │   │
│  ┌──────────────────────────────────────────────────────┐  │   │
│  │ FOR EACH LAYER (in reverse):                         │  │   │
│  │                                                       │  │   │
│  │  • Compute gradients (autograd, FP32)               │  │   │
│  │  • Quantize gradients (E5M2 - high dynamic range)   │  │   │
│  │  • Dequantize for accumulation                      │  │   │
│  │  • STE: gradient flows as identity                  │  │   │
│  │  • [TRACK: gradient scale, overflow events]         │  │   │
│  │                                                       │  │   │
│  └──────────────────────────────────────────────────────┘  │   │
│       │                                                     │   │
│       ▼                                                     │   │
│  ∂loss/∂weights (FP32, accumulated)                        │   │
│                                                              │   │
└──────────────────────────────────────────────────────────────┘  │
                              │                                   │
                              ▼                                   │
┌─────────────────────────────────────────────────────────────────┐
│ 4. OPTIMIZER STEP (with stiffness factor)                      │
│                                                                 │
│  • Gradient clipping (if enabled)                             │
│  • Apply stiffness factor: grad *= stiffness(iter_num)        │
│  • AdamW step:                                                │
│      - Update momentum (m_t = β1*m_{t-1} + (1-β1)*grad)       │
│      - Update variance (v_t = β2*v_{t-1} + (1-β2)*grad²)      │
│      - Compute update: Δw = -lr * m_t / (√v_t + ε)            │
│      - Apply weight decay                                      │
│  • Update weights (FP32 master copy)                          │
│  • Zero gradients                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. LOGGING & CHECKPOINTING                                     │
│                                                                 │
│  • Log loss, learning rate                                     │
│  • Log bit-tracking statistics (per-layer format usage)        │
│  • Log stiffness factor value                                  │
│  • Save checkpoint if validation loss improves                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight: Quantization is a Sandwich

Every quantized operation follows this pattern:
```
FP32 input → Quantize (FP32→FP8 sim) → Dequantize (FP8 sim→FP32) → FP32 compute
```

**Why?** PyTorch operations expect FP32/BF16. We simulate FP8 by:
1. Rounding to nearest representable FP8 value (in FP32)
2. Immediately converting back to FP32
3. Computing as normal

The "damage" is done by the rounding step, but gradients flow via STE.

---

## Scaling Factor Management

### Per-Tensor vs Per-Channel

| Strategy | Granularity | Memory Overhead | Accuracy | Use Case |
|----------|-------------|-----------------|----------|----------|
| **Per-Tensor** | 1 scale per tensor | 1 FP32 value | Good | Weights with uniform distribution |
| **Per-Channel** | 1 scale per output channel | N FP32 values (N=channels) | Better | Weights with outlier channels |
| **Per-Block (MX)** | 1 scale per 32 elements | High | Best | Blackwell GPUs only, not for H100 |

**Recommendation for test bench:** Start with **per-tensor static scaling** for weights, **per-tensor dynamic scaling** for activations.

### Dynamic vs Delayed Scaling

```python
# Dynamic (current) scaling - compute scale each forward pass
def dynamic_scale(tensor, format):
    amax = tensor.abs().max()
    fp8_max = format.max_representable_value()
    return fp8_max / amax

# Delayed scaling - use historical max (more stable)
def delayed_scale(tensor, format, history_window=100):
    amax = tensor.abs().max()
    history.append(amax)
    if len(history) > history_window:
        history.pop(0)
    amax_delayed = max(history)
    fp8_max = format.max_representable_value()
    return fp8_max / amax_delayed
```

**For research:** Test BOTH. Dynamic may reveal transient behavior; delayed may be more stable.

---

## Straight-Through Estimator (STE) Implementation

The STE is critical for gradient flow. Here's the canonical implementation:

```python
class FakeQuantize(torch.autograd.Function):
    """Quantize-dequantize with STE for gradients."""

    @staticmethod
    def forward(ctx, input, scale, format_spec):
        # Quantize: FP32 → FP8 simulation
        quantized = format_spec.quantize(input, scale)
        # Dequantize: FP8 simulation → FP32
        dequantized = format_spec.dequantize(quantized, scale)
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        # STE: treat quantization as identity for gradients
        return grad_output, None, None  # grad_input, grad_scale, grad_format

# Usage
fake_quant = FakeQuantize.apply
```

**What this does:**
- Forward: Simulates quantization rounding
- Backward: Passes gradient unchanged (∂output/∂input = 1)

**Why this works:** The STE approximates the gradient locally. Research (Yin et al., ICLR 2019) proves that with proper choice of STE, the "coarse gradient" correlates positively with the true population gradient.

**Alternative STEs to test:**
- Identity (standard): `grad_input = grad_output`
- Clipped identity: `grad_input = grad_output * (|input| <= threshold)`
- ReLU-based: `grad_input = grad_output * (input > 0)`

For FP8 research, **identity STE is standard**.

---

## Build Order and Dependencies

### Phase 1: Quantization Engine (Independent)
**Build first, no dependencies.**

1. Implement `FP8Format` class
   - Define E4M3, E5M2 standard formats
   - Add custom formats (E1M6, E3M4, E7M0)
   - Implement quantize/dequantize logic
2. Implement STE autograd function
3. Unit test with synthetic tensors
   - Test dynamic range coverage
   - Verify gradient flow (STE passthrough)
   - Test edge cases (overflow, underflow, NaN)

**Deliverable:** Standalone `quantization.py` module

**Validation:** Can quantize/dequantize arbitrary tensors with correct gradient flow.

---

### Phase 2: Wrapper Layers (Depends on Phase 1)
**Wrap standard nn.Linear with quantization.**

1. Implement `QuantizedLinear`
   - Takes original `nn.Linear` + format specs
   - Quantize weights (static scale) and activations (dynamic scale)
   - Forward pass returns FP32 after quant/dequant
2. Implement `quantize_model()` surgery function
   - Traverse model.named_modules()
   - Replace Linear layers in-place
3. Test on toy model (2-layer MLP)

**Deliverable:** `quantized_layers.py` module

**Validation:** Can replace layers in a simple model, forward/backward passes work.

---

### Phase 3: NanoGPT Integration (Depends on Phase 2)
**Minimal modification to nanoGPT.**

1. Copy nanoGPT's `model.py` to local project (or import)
2. Create initialization script:
   ```python
   model = GPT(config)
   model = quantize_model(model, layer_format_map)
   ```
3. Verify forward/backward pass on random input
4. Verify checkpoint save/load works

**Deliverable:** `initialize_model.py` script

**Validation:** Can load GPT-2 weights, apply quantization wrappers, and run inference.

---

### Phase 4: Training Loop Customization (Depends on Phase 3)
**Modify nanoGPT's train.py for custom optimizer.**

1. Copy `train.py` to local project
2. Replace optimizer creation:
   ```python
   optimizer = StiffnessAdamW(model.parameters(), lr, stiffness_factor)
   ```
3. Add bit-tracking hooks:
   ```python
   if iter_num % track_interval == 0:
       bit_tracker.track_all_layers(model)
       wandb.log(bit_tracker.report())
   ```
4. Test training loop on small dataset (1000 steps)

**Deliverable:** `train_fp8.py` script

**Validation:** Can train for 1000 steps, loss decreases, logs appear in WandB.

---

### Phase 5: Bit-Tracking System (Parallel with Phase 4)
**Can be developed in parallel.**

1. Implement `BitTracker` class
   - Convert FP8-simulated tensors to binary representation
   - Count bit usage per position
   - Detect overflow/underflow events
2. Integrate with wrapper layers:
   ```python
   # In QuantizedLinear.forward()
   self.last_quantized_weight = w_quant  # Cache for tracking
   ```
3. Create WandB dashboard for bit-level metrics

**Deliverable:** `bit_tracker.py` module

**Validation:** Can track bit usage, logs show mantissa vs exponent utilization.

---

### Phase 6: Dataset Integration (Independent)
**Can be developed in parallel with Phases 1-3.**

1. Download EurLex dataset
2. Tokenize with GPT-2 tokenizer
3. Create `train.bin` / `val.bin` following nanoGPT format
4. Test data loading in nanoGPT's `train.py`

**Deliverable:** `data/eurlex/prepare.py` script

**Validation:** Can load batches, tokens are valid, dataset size < 1M tokens.

---

### Phase 7: Full Integration Test
**All components together.**

1. Run full training loop:
   - Quantized model (mixed precision per layer)
   - Custom optimizer with stiffness factor
   - Bit-tracking enabled
   - WandB logging
2. Train for 5000 steps on EurLex
3. Compare against baseline (no quantization)
4. Generate comparison report

**Deliverable:** Working test bench

**Validation:** Can train to convergence, bit-tracking data shows format differences, results are reproducible.

---

## Architectural Patterns to Follow

### Pattern 1: Separation of Concerns
**What:** Keep quantization logic separate from model definition.
**Why:** Enables easy experimentation with different formats without modifying nanoGPT.
**How:** Use wrapper layers and model surgery instead of forking nanoGPT.

### Pattern 2: Lazy Scale Computation
**What:** Compute weight scales once (static), activation scales per-batch (dynamic).
**Why:** Weights are static (scale once at initialization or first forward pass), activations vary per input.
**How:**
```python
if self.weight_scale is None:  # Compute once
    self.weight_scale = compute_scale(self.weight)
act_scale = compute_scale(activations)  # Compute every forward
```

### Pattern 3: Instrumentation Without Overhead
**What:** Add tracking hooks that can be disabled via config flag.
**Why:** Bit-tracking adds overhead; only enable when needed.
**How:**
```python
if config.enable_tracking and iter_num % config.track_interval == 0:
    bit_tracker.track(...)
```

### Pattern 4: Format Polymorphism
**What:** All FP8 formats share same interface (`quantize`, `dequantize`, `compute_scale`).
**Why:** Easy to swap formats in experiments.
**How:** Abstract base class or protocol:
```python
class QuantizationFormat(Protocol):
    def quantize(self, tensor, scale) -> Tensor: ...
    def dequantize(self, tensor, scale) -> Tensor: ...
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Forking NanoGPT
**Why bad:** Merge conflicts, maintenance burden, hard to update to new nanoGPT versions.
**Instead:** Use model surgery (replace layers post-initialization).

### Anti-Pattern 2: Quantizing Embeddings
**Why bad:** Embeddings are lookup tables, not arithmetic-intensive. Quantizing adds complexity for no benefit.
**Instead:** Only quantize Linear layers (attention, MLP, classifier).

### Anti-Pattern 3: Per-Tensor Scaling for Activations
**Why bad:** Activations vary widely per batch; static scaling causes overflow/underflow.
**Instead:** Use dynamic per-tensor scaling (compute scale each forward pass).

### Anti-Pattern 4: Ignoring Gradient Quantization
**Why bad:** Forward-only quantization misses half the story; gradients use different precision needs.
**Instead:** Quantize gradients with E5M2 (higher dynamic range).

### Anti-Pattern 5: Hardcoding Format Specs
**Why bad:** Makes it hard to experiment with E1M6, E3M4, E7M0.
**Instead:** Use configuration files or CLI args:
```python
layer_formats = {
    'attention': FP8Format.from_string(config.attention_format),  # "E4M3"
    'mlp': FP8Format.from_string(config.mlp_format)  # "E3M4"
}
```

---

## Scalability Considerations

### At Current Scale (≤50M params, single H100)

| Component | Approach | Rationale |
|-----------|----------|-----------|
| **Quantization** | Per-tensor scaling | Sufficient for single-GPU, low overhead |
| **Bit tracking** | Sample every 100 steps | Reduces logging overhead |
| **Dataset** | In-memory loading | EurLex is small, fits in RAM |
| **Checkpointing** | Every 1000 steps | Fast SSD on H100 node |

### If Scaling to Multi-GPU (Future)

| Component | Change Needed | Notes |
|-----------|---------------|-------|
| **Quantization** | Per-channel or MX formats | Better accuracy at scale |
| **Bit tracking** | Distributed gather | Aggregate stats across GPUs |
| **Dataset** | Memory-mapped arrays | Avoid RAM limits |
| **Optimizer** | Distributed wrapper | Muon has distributed overhead handling |

**For this project:** Single-GPU is sufficient. No need to optimize for multi-GPU yet.

---

## Sources

### High Confidence (Context7 / Official Documentation)
- [NVIDIA Transformer Engine FP8 Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [PyTorch FakeQuantize API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)
- [PyTorch QAT Tutorial](https://pytorch.org/blog/quantization-aware-training/)
- [Understanding FP8 Format (Scaleway)](https://www.scaleway.com/en/docs/gpu/reference-content/understanding-nvidia-fp8/)

### Medium Confidence (Verified Research Papers)
- [Per-Tensor and Per-Block Scaling Strategies (NVIDIA Blog)](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [Understanding Straight-Through Estimator (Yin et al., ICLR 2019)](https://arxiv.org/abs/1903.05662)
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/pdf/2110.02861)
- [Mixed-Precision Quantization for LLMs](https://arxiv.org/html/2510.16805v1)
- [Effective Quantization of Muon Optimizer States](https://arxiv.org/html/2509.23106v1)

### Low Confidence (WebSearch Only, Community Knowledge)
- [NanoGPT Repository Structure](https://github.com/karpathy/nanoGPT)
- [Modded-NanoGPT Custom Optimizer Patterns](https://github.com/KellerJordan/modded-nanogpt)
- [WandB PyTorch Integration](https://docs.wandb.ai/tutorials/experiments/)

### Key References for Roadmap Planning
- [Taxonomy of Small Floating-Point Formats (UW PLSE, Feb 2025)](https://uwplse.org/2025/02/17/Small-Floats.html) - Lists non-standard formats
- [Microscaling Floating Point Formats for LLMs](https://arxiv.org/html/2510.01863v1) - MXFP8 background
- [Mix-QViT Layer Importance Analysis](https://arxiv.org/html/2501.06357v1) - Per-layer sensitivity methodology

---

## Confidence Assessment

| Area | Confidence | Reasoning |
|------|------------|-----------|
| **Quantization mechanics** | HIGH | Official NVIDIA docs, PyTorch source code |
| **STE implementation** | HIGH | Research paper (Yin et al.) + PyTorch examples |
| **NanoGPT integration** | MEDIUM | Based on code structure, not explicit QAT examples |
| **Bit-tracking approach** | MEDIUM | Novel requirement, no standard library support |
| **Stiffness optimizer** | LOW | Custom requirement, limited prior art |
| **Build order** | HIGH | Standard dependency graph analysis |

---

## Open Questions for Phase-Specific Research

1. **Stiffness Factor Implementation:** How exactly should stiffness modulate optimizer step? Gradient scaling vs learning rate scaling vs momentum decay?

2. **Bit-Tracking Granularity:** Track per-layer, per-tensor, or per-element? Memory vs insight tradeoff?

3. **E1M6 / E7M0 Edge Cases:** These non-standard formats may have numerical instability. Need empirical testing.

4. **Validation Metrics:** Beyond loss, what metrics indicate "good" quantization? Gradient SNR? Weight update magnitude?

5. **Baseline Comparison:** Train identical model without quantization for apples-to-apples comparison?

---

## Ready for Roadmap Creation

This architecture research provides:
- ✅ Clear component boundaries
- ✅ Data flow with quantization points identified
- ✅ Build order with dependency analysis
- ✅ Integration strategy for nanoGPT
- ✅ Custom optimizer integration approach
- ✅ Bit-tracking system design

**Next step:** Use this to structure roadmap phases, prioritizing Quantization Engine (independent) → Wrapper Layers → NanoGPT Integration → Full System.
