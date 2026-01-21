---
phase: 03-model-integration
plan: 02
subsystem: integration
tags: [quantization, mixed-precision, reproducibility, config, seed]

dependency_graph:
  requires: ["03-01"]
  provides: ["per-layer-precision", "reproducible-ablation"]
  affects: ["04-experiment-runs"]

tech_stack:
  added: []
  patterns: ["per-layer-config", "regex-pattern-matching", "rng-seeding"]

files:
  created:
    - altgrad/integration/config.py
    - altgrad/utils/__init__.py
    - altgrad/utils/reproducibility.py
    - tests/test_reproducibility.py
  modified:
    - altgrad/integration/surgery.py
    - altgrad/integration/__init__.py
    - tests/test_integration.py

decisions:
  - key: first-match-wins
    choice: First matching LayerPrecisionRule determines format
    rationale: Allows specific patterns to override general ones
  - key: none-means-bf16
    choice: format=None in rules means keep layer in BF16
    rationale: Enables mixed precision without special "BF16" format type
  - key: mutex-format-config
    choice: quantize_model requires exactly one of format or config
    rationale: Clear API - single format mode vs per-layer config mode

metrics:
  duration: 6 min
  completed: 2026-01-21
---

# Phase 03 Plan 02: Mixed Precision Config and Reproducibility Summary

Per-layer mixed precision configuration via regex pattern matching, plus comprehensive seed setup for reproducible ablation experiments.

## One-Liner

QuantizationConfig routes layers to FP8 formats via regex (attention BF16, MLP E5M2), with set_seed_for_reproducibility() ensuring identical initial states across ablation runs.

## What Was Built

### QuantizationConfig (altgrad/integration/config.py - 185 lines)

Per-layer precision configuration using regex pattern matching:

```python
from altgrad.integration import QuantizationConfig, LayerPrecisionRule

config = QuantizationConfig(
    default_format="E5M2",
    layer_rules=[
        LayerPrecisionRule(r"\.attn\.", None),     # Attention stays BF16
        LayerPrecisionRule(r"\.mlp\.", "E5M2"),    # MLP uses FP8
        LayerPrecisionRule(r"lm_head", None),      # LM head stays BF16
    ],
)

# First matching rule wins
format = config.get_format_for_layer("transformer.h.0.mlp.c_fc")  # E5M2
format = config.get_format_for_layer("transformer.h.0.attn.c_attn")  # None (BF16)
```

Key features:
- `LayerPrecisionRule`: Regex pattern + format (None = BF16)
- `QuantizationConfig.get_format_for_layer()`: First match wins semantics
- `create_mixed_precision_config()`: Helper for common GPT patterns

### Updated Surgery (altgrad/integration/surgery.py)

`quantize_model()` now supports two modes:

```python
# Mode 1: Single format (existing behavior)
quantize_model(model, E5M2, skip_patterns=["lm_head"])

# Mode 2: Per-layer config (new)
config = create_mixed_precision_config(attention_format=None, mlp_format="E5M2")
quantize_model(model, config=config)
```

Backward compatible - existing single-format calls still work.

### Reproducibility Utilities (altgrad/utils/reproducibility.py - 223 lines)

Comprehensive seed setup for ablation experiments:

```python
from altgrad.utils import set_seed_for_reproducibility

# Ablation run 1: E5M2
set_seed_for_reproducibility(42)
model1 = GPT(config)
quantize_model(model1, E5M2)

# Ablation run 2: E3M4 (same seed = identical initial weights)
set_seed_for_reproducibility(42)
model2 = GPT(config)
quantize_model(model2, E3M4)

# Only difference is quantization format
```

Sets all RNG sources:
- Python `random` module
- NumPy `np.random`
- PyTorch CPU and CUDA
- cuDNN deterministic mode
- CUBLAS workspace config

Additional utilities:
- `seed_worker()`: DataLoader worker seeding
- `create_reproducible_dataloader()`: Reproducible batch ordering
- `get_rng_state()` / `set_rng_state()`: Checkpoint/restore RNG state

## Key Patterns

### Pattern Matching Priority

Rules are evaluated in order; first match wins:
```python
rules=[
    LayerPrecisionRule(r"c_proj", "E3M4"),  # More specific - evaluated first
    LayerPrecisionRule(r"\.mlp\.", "E5M2"), # More general - fallback
]
# "mlp.c_proj" matches c_proj rule -> E3M4
# "mlp.c_fc" matches mlp rule -> E5M2
```

### Format = None Means BF16

No special BF16 format type needed:
```python
LayerPrecisionRule(r"\.attn\.", None)  # Attention stays nn.Linear (BF16)
```

When `get_format_for_layer()` returns None, `quantize_model()` skips that layer.

## Test Coverage

52 tests total (36 integration + 16 reproducibility):

**Config tests (10):**
- LayerPrecisionRule pattern matching
- QuantizationConfig format routing
- First-match-wins semantics
- Default format fallback
- quantize_model with config mode
- Error cases (neither/both format and config)

**Reproducibility tests (16):**
- Torch/NumPy/Python RNG seeding
- Different seeds produce different results
- GPT model initialization reproducibility
- Ablation same-seed-different-format validation
- DataLoader reproducibility
- RNG state capture/restore

## Deviations from Plan

None - plan executed exactly as written.

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `altgrad/integration/config.py` | 185 | Per-layer precision config |
| `altgrad/integration/surgery.py` | +44 | Config support in quantize_model |
| `altgrad/utils/reproducibility.py` | 223 | Seed setup for ablations |
| `altgrad/utils/__init__.py` | 29 | Utils module exports |
| `tests/test_integration.py` | +172 | Config tests |
| `tests/test_reproducibility.py` | 298 | Reproducibility tests |

## Next Phase Readiness

Ready for Phase 4 (Experiment Runs):
- Mixed precision enables attention-BF16/MLP-FP8 experiments
- Reproducibility ensures fair format comparisons
- Same seed guarantees identical initial weights across ablations

Blockers: None
