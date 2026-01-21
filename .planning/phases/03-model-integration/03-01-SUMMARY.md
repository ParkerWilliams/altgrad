---
phase: 03-model-integration
plan: 01
subsystem: integration
tags: [quantization, pytorch, nn.Linear, model-surgery, wrapper]

# Dependency graph
requires:
  - phase: 01-quantization-engine
    provides: FP8 formats, quantize function, AmaxHistory, compute_scale
provides:
  - QuantizedLinear wrapper for FP8 quantization at layer boundaries
  - quantize_model surgery function for injecting quantization
  - dequantize_model for restoring original modules
  - Weight tying preservation support
affects: [03-02-shadow-copy, 03-03-gradient-pipeline, 04-geometry-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns: [model-surgery, wrapper-pattern, skip-patterns-for-weight-tying]

key-files:
  created:
    - altgrad/integration/__init__.py
    - altgrad/integration/wrapper.py
    - altgrad/integration/surgery.py
    - tests/test_integration.py
  modified: []

key-decisions:
  - "Skip patterns approach for weight tying preservation (skip lm_head)"
  - "In-place surgery modifies model directly (no copy)"
  - "Collect-then-modify pattern to avoid mutating during iteration"
  - "QuantizedLinear exposes weight/bias properties for optimizer compatibility"

patterns-established:
  - "Model surgery: quantize_model(model, format, skip_patterns=['lm_head'])"
  - "Wrapper pattern: QuantizedLinear wraps nn.Linear, applies quantize() in forward"
  - "Nested module access via dotted path names"

# Metrics
duration: 3min
completed: 2026-01-21
---

# Phase 03 Plan 01: Model Integration Wrapper Summary

**QuantizedLinear wrapper with STE gradient flow and model surgery for injecting FP8 quantization into GPT models without modifying source code**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-21T22:40:31Z
- **Completed:** 2026-01-21T22:43:54Z
- **Tasks:** 3 (TDD workflow: RED-GREEN-GREEN)
- **Files created:** 4
- **Lines of code:** 701

## Accomplishments

- QuantizedLinear wrapper that applies FP8 quantization on forward pass with STE gradient flow
- Model surgery functions (quantize_model, dequantize_model) for post-construction quantization injection
- Weight tying preservation between GPT's wte and lm_head via skip_patterns
- 26 comprehensive tests covering wrapper, surgery, nested modules, and GPT integration

## Task Commits

Each task was committed atomically (TDD workflow):

1. **Task 1: Write failing tests for QuantizedLinear and surgery** - `a85637c` (test)
2. **Task 2: Implement QuantizedLinear wrapper** - `19c34dc` (feat)
3. **Task 3: Implement model surgery functions** - `b364406` (feat)

## Files Created

- `altgrad/integration/__init__.py` - Package exports (QuantizedLinear, quantize_model, dequantize_model)
- `altgrad/integration/wrapper.py` - QuantizedLinear wrapper with STE gradient flow (147 lines)
- `altgrad/integration/surgery.py` - Model surgery functions for quantization injection (171 lines)
- `tests/test_integration.py` - Comprehensive integration tests (383 lines, 26 tests)

## Decisions Made

1. **Skip patterns for weight tying:** Using skip_patterns=['lm_head'] to preserve weight tying between wte and lm_head in GPT models. This is simpler than tracking shared weights explicitly.

2. **In-place surgery:** Model surgery modifies the model directly rather than creating a copy. This matches PyTorch conventions and avoids memory overhead.

3. **Collect-then-modify pattern:** Surgery functions collect modules to replace first, then modify. This avoids the "dictionary changed size during iteration" error when using named_modules().

4. **Property exposure:** QuantizedLinear exposes weight/bias properties that delegate to the underlying Linear. This ensures optimizer parameter groups work correctly.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - implementation was straightforward following the TDD approach.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- QuantizedLinear and surgery functions are ready for use
- Next plan (03-02) can build FP32 shadow model that tracks FP32 copies of quantized weights
- Integration with existing GPT model verified working

---
*Phase: 03-model-integration*
*Completed: 2026-01-21*
