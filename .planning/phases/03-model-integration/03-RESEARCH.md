# Phase 3: Model Integration - Research

**Researched:** 2026-01-21
**Domain:** PyTorch module wrapping, model surgery, mixed-precision configuration
**Confidence:** HIGH

## Summary

Phase 3 integrates the existing quantization engine (Phase 1) with the GPT model (Phase 2) via QuantizedLinear wrappers and a quantize_model() surgery function. The standard approach is:

1. **QuantizedLinear wrapper** - A module that wraps nn.Linear, applying FP8 quantization on forward pass using the existing `quantize()` function and `AmaxHistory` for dynamic scaling. The wrapper inherits from nn.Module (not nn.Linear) to maintain clear separation between quantization logic and linear computation.

2. **quantize_model() surgery** - A recursive function that traverses the model's module tree and replaces nn.Linear layers with QuantizedLinear, preserving weights and configurations. Uses the `setattr` pattern with parent module access via `named_modules()`.

3. **Per-layer mixed precision config** - A dataclass-based configuration specifying which layer patterns get which FP8 format (or None for BF16). Layer matching via name patterns (e.g., "attn" vs "mlp").

4. **Format ablation runs** - Use identical seeds with comprehensive RNG state management; the only variable is the FP8 format.

**Primary recommendation:** Create QuantizedLinear as a thin wrapper that delegates to the original nn.Linear but intercepts forward pass to apply quantize/dequantize. Use a LayerPrecisionConfig dataclass with layer name patterns for per-layer format assignment.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch nn.Module | 2.x | Base class for wrapper | Native PyTorch pattern |
| altgrad.quantization | existing | quantize(), AmaxHistory, FP8Format | Already implemented in Phase 1 |
| dataclasses | stdlib | LayerPrecisionConfig | Clean configuration objects |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| copy.deepcopy | stdlib | Clone model before surgery | When original model must be preserved |
| functools.reduce | stdlib | Recursive getattr for nested modules | Deep module access |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom wrapper | torch.ao.nn.quantized.Linear | PyTorch's version is for inference quantization, not training simulation |
| setattr surgery | monkey-patching quant_modules | Too invasive, affects global state |
| Pattern matching | torch.fx graph rewriting | Overkill for simple layer replacement |

**Installation:**
No additional dependencies - all tools are in PyTorch stdlib or existing altgrad.quantization module.

## Architecture Patterns

### Recommended Project Structure
```
altgrad/
├── quantization/          # Existing from Phase 1
│   ├── __init__.py
│   ├── formats.py
│   ├── ops.py
│   └── scaling.py
├── integration/           # NEW for Phase 3
│   ├── __init__.py
│   ├── wrapper.py         # QuantizedLinear
│   ├── surgery.py         # quantize_model(), dequantize_model()
│   └── config.py          # LayerPrecisionConfig, QuantizationConfig
└── training/              # Existing from Phase 2
    ├── model.py
    └── trainer.py
```

### Pattern 1: QuantizedLinear Wrapper

**What:** A wrapper module that contains an nn.Linear and applies quantization during forward pass.

**When to use:** All Linear layers that should use simulated FP8 quantization.

**Example:**
```python
# Based on TorchAO pattern from official PyTorch docs
class QuantizedLinear(nn.Module):
    """Linear layer with simulated FP8 quantization.

    Wraps an existing nn.Linear and applies quantize/dequantize
    on forward pass. Maintains AmaxHistory for dynamic scaling.
    """

    def __init__(
        self,
        linear: nn.Linear,
        fp8_format: FP8Format,
        history_len: int = 16,
    ):
        super().__init__()
        self.linear = linear  # Keep original linear
        self.fp8_format = fp8_format
        self.weight_history = AmaxHistory(history_len)
        self.input_history = AmaxHistory(history_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Update amax histories
        self.weight_history.update(self.linear.weight)
        self.input_history.update(x)

        # Compute scales
        w_scale = compute_scale(self.weight_history.get_amax(), self.fp8_format)
        x_scale = compute_scale(self.input_history.get_amax(), self.fp8_format)

        # Quantize weight and input
        w_scale_t = torch.tensor(w_scale, device=x.device, dtype=x.dtype)
        x_scale_t = torch.tensor(x_scale, device=x.device, dtype=x.dtype)

        w_q = quantize(self.linear.weight, self.fp8_format, w_scale_t)
        x_q = quantize(x, self.fp8_format, x_scale_t)

        # Compute output (STE handles gradients)
        out = F.linear(x_q, w_q, self.linear.bias)
        return out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        fp8_format: FP8Format,
        history_len: int = 16,
    ) -> "QuantizedLinear":
        """Create QuantizedLinear from existing nn.Linear."""
        return cls(linear, fp8_format, history_len)

    @property
    def weight(self) -> torch.Tensor:
        """Access wrapped linear's weight for compatibility."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Access wrapped linear's bias for compatibility."""
        return self.linear.bias
```

### Pattern 2: Recursive Module Surgery

**What:** A function that traverses the model tree and replaces matching modules.

**When to use:** After model initialization, before training begins.

**Example:**
```python
# Based on PyTorch Forum discussions and TorchAO patterns
def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    inplace: bool = True,
) -> nn.Module:
    """Replace Linear layers with QuantizedLinear based on config.

    Args:
        model: Model to quantize
        config: Configuration specifying per-layer formats
        inplace: Modify model in place (False creates deepcopy)

    Returns:
        Model with Linear layers replaced by QuantizedLinear
    """
    if not inplace:
        model = copy.deepcopy(model)

    # Build dict for parent access
    modules_dict = dict(model.named_modules())

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Get format for this layer (None means skip/BF16)
            fp8_format = config.get_format_for_layer(name)

            if fp8_format is not None:
                # Create quantized wrapper
                quantized = QuantizedLinear.from_linear(
                    module, fp8_format, config.history_len
                )

                # Replace in parent
                _set_module_by_name(model, name, quantized)

    return model


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Set a module by its dot-separated name path."""
    parts = name.rsplit(".", 1)
    if len(parts) == 1:
        # Top-level module
        setattr(model, name, new_module)
    else:
        # Nested module
        parent_name, child_name = parts
        parent = dict(model.named_modules())[parent_name]
        setattr(parent, child_name, new_module)
```

### Pattern 3: Layer Precision Configuration

**What:** A dataclass-based config that maps layer name patterns to FP8 formats.

**When to use:** To specify which layers use FP8 and which stay in BF16.

**Example:**
```python
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import re


@dataclass
class LayerPrecisionRule:
    """Rule for matching layer names to FP8 formats."""
    pattern: str  # Regex pattern for layer name
    format: Optional[str]  # FP8 format name (None = skip/BF16)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        default_format: Default FP8 format for unmatched layers (None = BF16)
        layer_rules: Ordered list of rules (first match wins)
        history_len: Amax history length for dynamic scaling
        quantize_embeddings: Whether to quantize embedding layers
    """
    default_format: Optional[str] = "E5M2"
    layer_rules: List[LayerPrecisionRule] = field(default_factory=list)
    history_len: int = 16
    quantize_embeddings: bool = False

    def get_format_for_layer(self, layer_name: str) -> Optional[FP8Format]:
        """Get FP8 format for a layer based on rules.

        Returns None if layer should stay in BF16.
        """
        for rule in self.layer_rules:
            if re.search(rule.pattern, layer_name):
                if rule.format is None:
                    return None  # BF16
                return FORMAT_REGISTRY[rule.format]

        # Use default
        if self.default_format is None:
            return None
        return FORMAT_REGISTRY[self.default_format]


# Example: Attention in BF16, MLP in FP8
def create_mixed_precision_config() -> QuantizationConfig:
    """Create config with attention in BF16, MLP in FP8."""
    return QuantizationConfig(
        default_format="E5M2",  # Default to FP8
        layer_rules=[
            LayerPrecisionRule(r"\.attn\.", None),    # Attention stays BF16
            LayerPrecisionRule(r"\.c_attn", None),    # QKV projection BF16
            LayerPrecisionRule(r"\.c_proj", "E5M2"),  # Output projection FP8
            LayerPrecisionRule(r"\.mlp\.", "E5M2"),   # MLP layers FP8
            LayerPrecisionRule(r"lm_head", None),     # LM head BF16 (sensitive)
        ],
    )
```

### Anti-Patterns to Avoid

- **Modifying nn.Linear class globally:** Never monkey-patch nn.Linear itself. Always create a separate wrapper class.

- **In-place weight mutation during forward:** The current Trainer pattern temporarily modifies weights. Better to wrap the module so original weights stay untouched and quantized versions are computed fresh each forward.

- **Breaking weight tying:** GPT uses weight tying between embedding and lm_head. When replacing lm_head, must preserve the weight sharing relationship.

- **Iterating and mutating:** Never modify `named_modules()` while iterating. Build list first, then modify.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Recursive getattr | Manual loop | `functools.reduce` with getattr | Handles arbitrary depth cleanly |
| Layer name matching | Exact string match | `re.search` with patterns | More flexible, handles variations |
| RNG state save/restore | Manual tensor copies | PyTorch checkpoint functions | Already handles all RNG sources |
| Model copying | Manual parameter copy | `copy.deepcopy` | Handles nested modules, buffers |

**Key insight:** PyTorch's module system is well-designed for wrapping and surgery. Use its patterns (named_modules, setattr) rather than custom traversal.

## Common Pitfalls

### Pitfall 1: Breaking Weight Tying in GPT

**What goes wrong:** After surgery, lm_head and wte.weight are no longer the same tensor, breaking weight tying.

**Why it happens:** Wrapping lm_head creates a new QuantizedLinear with its own copy of the weight.

**How to avoid:** Either skip lm_head quantization (common practice - LM head is sensitive), or explicitly re-establish weight tying after surgery:
```python
if hasattr(model, 'lm_head') and hasattr(model.transformer, 'wte'):
    model.lm_head.linear.weight = model.transformer.wte.weight
```

**Warning signs:** Model produces garbage after quantization; embedding and output logits are inconsistent.

### Pitfall 2: RuntimeError During Module Iteration

**What goes wrong:** `RuntimeError: OrderedDict mutated during iteration` when using `named_modules()`.

**Why it happens:** Calling `setattr` while iterating modifies the module tree.

**How to avoid:** Collect modules to replace first with `list(model.named_modules())`, then iterate over the list to make replacements.

**Warning signs:** Exception raised during quantize_model().

### Pitfall 3: Lost Gradient Flow with Wrapper

**What goes wrong:** Gradients don't flow through QuantizedLinear wrapper.

**Why it happens:** If wrapper doesn't use autograd-compatible operations or breaks the computational graph.

**How to avoid:**
1. Use `quantize()` which already implements STE via QuantizeFunc autograd function
2. Don't detach tensors or use `.data` directly
3. Keep wrapper's forward purely functional

**Warning signs:** Zero gradients on original linear weights; model doesn't learn.

### Pitfall 4: Inconsistent Scaling Across Forward/Backward

**What goes wrong:** Quantization uses different scales in forward vs backward, causing training instability.

**Why it happens:** AmaxHistory updated during forward, but backward uses stale scale.

**How to avoid:** The STE pattern means backward uses same quantized values. Scale is computed once per forward pass and the quantize() function handles gradient passthrough correctly.

**Warning signs:** Training loss oscillates wildly; gradients spike.

### Pitfall 5: Non-Reproducible Ablation Runs

**What goes wrong:** Runs with different FP8 formats produce different random sequences beyond quantization effects.

**Why it happens:** Not seeding all RNG sources; DataLoader worker seeds not controlled.

**How to avoid:** Comprehensive seed setup:
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
Plus DataLoader worker seeding with `worker_init_fn` and `generator`.

**Warning signs:** Different loss curves for identical configs on same data.

## Code Examples

Verified patterns from official sources:

### Complete QuantizedLinear with Gradient Test
```python
# Pattern verified against TorchAO documentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from altgrad.quantization import quantize, compute_scale, AmaxHistory, FP8Format


class QuantizedLinear(nn.Module):
    """Simulated FP8 Linear layer for QAT."""

    def __init__(self, linear: nn.Linear, fp8_format: FP8Format):
        super().__init__()
        self.linear = linear
        self.fp8_format = fp8_format
        self.weight_history = AmaxHistory()
        self.input_history = AmaxHistory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic scale computation
        self.weight_history.update(self.linear.weight.detach())
        self.input_history.update(x.detach())

        w_scale = compute_scale(self.weight_history.get_amax(), self.fp8_format)
        x_scale = compute_scale(self.input_history.get_amax(), self.fp8_format)

        # Quantize (STE handles gradients)
        w_scale_t = torch.tensor(w_scale, device=x.device, dtype=x.dtype)
        x_scale_t = torch.tensor(x_scale, device=x.device, dtype=x.dtype)

        w_q = quantize(self.linear.weight, self.fp8_format, w_scale_t)
        x_q = quantize(x, self.fp8_format, x_scale_t)

        return F.linear(x_q, w_q, self.linear.bias)


# Test gradient flow
def test_gradient_flow():
    linear = nn.Linear(10, 5)
    from altgrad.quantization import E5M2
    q_linear = QuantizedLinear(linear, E5M2)

    x = torch.randn(4, 10, requires_grad=True)
    y = q_linear(x)
    y.sum().backward()

    assert x.grad is not None, "Input gradient should exist"
    assert linear.weight.grad is not None, "Weight gradient should exist"
    print("Gradient flow test passed!")
```

### Recursive Surgery Function
```python
# Pattern from PyTorch Forum: https://discuss.pytorch.org/t/replacing-layers/60068
def quantize_model(
    model: nn.Module,
    fp8_format: FP8Format,
    skip_patterns: list[str] = None,
) -> nn.Module:
    """Replace Linear layers with QuantizedLinear.

    Args:
        model: Model to modify (in-place)
        fp8_format: FP8 format for quantization
        skip_patterns: Layer name patterns to skip
    """
    skip_patterns = skip_patterns or []

    # Collect replacements first (avoid mutating during iteration)
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check skip patterns
            should_skip = any(p in name for p in skip_patterns)
            if not should_skip:
                replacements.append((name, module))

    # Apply replacements
    for name, module in replacements:
        q_module = QuantizedLinear(module, fp8_format)

        # Navigate to parent and replace
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            setattr(model, name, q_module)
        else:
            parent_name, child_name = parts
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, q_module)

    return model
```

### Comprehensive Seed Setup for Ablations
```python
# Pattern from PyTorch Reproducibility docs
import os
import random
import numpy as np
import torch


def set_seed_for_reproducibility(seed: int):
    """Set all random seeds for reproducible experiments.

    Source: https://pytorch.org/docs/stable/notes/randomness.html
    """
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # CUBLAS (for CUDA >= 10.2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int):
    """DataLoader worker seed function."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(dataset, batch_size: int, seed: int):
    """Create DataLoader with reproducible shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Global quant_modules.initialize() | Per-layer QuantizedLinear wrappers | ~2024 | Finer control, no global state |
| Manual weight quantization in Trainer | QuantizedLinear encapsulates logic | Phase 3 | Cleaner separation of concerns |
| Static scales | Dynamic per-tensor scaling with AmaxHistory | Already in Phase 1 | Better range utilization |

**Deprecated/outdated:**
- `torch.ao.nn.quantized.Linear`: Designed for inference, not QAT
- `pytorch-quantization.quant_modules.initialize()`: Too coarse-grained

## Open Questions

Things that couldn't be fully resolved:

1. **Whether to quantize both input and weight or just weight**
   - What we know: Production FP8 typically quantizes both (FP8-FP8 matmul)
   - What's unclear: For simulated FP8 without tensor cores, input quantization adds overhead
   - Recommendation: Start with weight-only, add input quantization as option

2. **Handling of LayerNorm and Embeddings**
   - What we know: These are typically kept in higher precision (BF16/FP32)
   - What's unclear: Exact sensitivity for 10M param model
   - Recommendation: Exclude from quantization initially; config option to include

3. **Integration with existing Trainer vs. standalone**
   - What we know: Current Trainer has its own quantization loop in `_quantized_forward_context`
   - What's unclear: Whether to deprecate that in favor of QuantizedLinear
   - Recommendation: QuantizedLinear for Phase 3, keep Trainer's approach as fallback

## Sources

### Primary (HIGH confidence)
- [TorchAO Subclass Basic Documentation](https://docs.pytorch.org/ao/stable/subclass_basic.html) - QuantizedLinear pattern, from_float classmethod
- [PyTorch Reproducibility Guide](https://docs.pytorch.org/docs/stable/notes/randomness.html) - Seed setup, cudnn deterministic settings
- altgrad/quantization (existing codebase) - quantize(), AmaxHistory, FP8Format APIs

### Secondary (MEDIUM confidence)
- [PyTorch Forum: Replacing Layers](https://discuss.pytorch.org/t/replacing-layers-in-model-with-named-modules/124925) - Module surgery patterns
- [PyTorch Practical Quantization Blog](https://pytorch.org/blog/quantization-in-practice/) - Workflow patterns
- [NVIDIA Transformer Engine Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) - Per-layer precision concepts

### Tertiary (LOW confidence - needs validation)
- [Megatron Bridge Mixed Precision](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/mixed-precision.html) - MixedPrecisionConfig patterns
- WebSearch results on ablation study reproducibility - General best practices

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses only PyTorch built-ins and existing altgrad modules
- Architecture: HIGH - Based on official PyTorch/TorchAO patterns
- Surgery pattern: HIGH - Well-documented in PyTorch forums
- Per-layer config: MEDIUM - Custom design, but follows common patterns
- Pitfalls: HIGH - Based on actual PyTorch behavior and common errors
- Ablation reproducibility: HIGH - Official PyTorch documentation

**Research date:** 2026-01-21
**Valid until:** 2026-02-21 (30 days - patterns are stable)
