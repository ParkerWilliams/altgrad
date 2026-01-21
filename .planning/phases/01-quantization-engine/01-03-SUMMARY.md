---
phase: 01-quantization-engine
plan: 03
subsystem: quantization
tags: [pytorch, fp8, quantization, dynamic-scaling, diagnostics, amax-history, bit-stall]

# Dependency graph
requires:
  - phase: 01-01
    provides: FP8 format registry with transfer functions
  - phase: 01-02
    provides: Quantize/dequantize operations with STE gradient
provides:
  - Per-tensor dynamic scaling with amax history smoothing
  - Bit-stall detection for identifying format precision issues
  - Complete quantization package exports with unified API
affects: [01-04-quantized-layers, 01-05-format-testing, 02-geometry-aware-optimization]

# Tech tracking
tech-stack:
  added: [collections.deque]
  patterns: [sliding-window-amax-tracking, stall-detection-via-quantization-comparison]

key-files:
  created:
    - altgrad/quantization/scaling.py
    - altgrad/quantization/diagnostics.py
    - tests/test_scaling.py
    - tests/test_diagnostics.py
  modified:
    - altgrad/quantization/__init__.py
    - altgrad/__init__.py

key-decisions:
  - "Use max of history (not moving average) for stable scaling"
  - "Detect stall via quantize comparison rather than ULP calculation"
  - "Scale_min=1e-10 prevents division by zero on all-zero tensors"
  - "Count only non-zero gradients in stall detection (grad.abs() > 1e-10)"

patterns-established:
  - "AmaxHistory with deque for sliding-window tracking"
  - "BitStallDetector accumulates statistics across training steps"
  - "Package exports all public APIs from submodule __init__.py"

# Metrics
duration: 5min
completed: 2026-01-21
---

# Phase 01 Plan 03: Scaling and Diagnostics Summary

**Per-tensor dynamic scaling with 16-batch amax history and bit-stall detection for identifying format precision limitations**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-21T15:34:03Z
- **Completed:** 2026-01-21T15:39:34Z
- **Tasks:** 3
- **Files modified:** 6
- **Tests:** 111 total (34 new)

## Accomplishments

- Dynamic scaling maps tensor ranges to FP8 representable ranges via amax history
- Bit-stall detection identifies when quantized gradients are too small to update weights
- Complete altgrad.quantization package with formats, operations, scaling, and diagnostics
- All components tested and integrated with 100% test pass rate

## Task Commits

Each task was committed atomically:

1. **Task 1: Per-tensor scaling with amax history** - `ca4fb66` (feat)
2. **Task 2: Bit-stall detection** - `a585bbe` (feat)
3. **Task 3: Package integration and exports** - `c812ad9` (feat)

## Files Created/Modified

### Created
- `altgrad/quantization/scaling.py` - Dynamic scaling with AmaxHistory class and compute_scale function
- `altgrad/quantization/diagnostics.py` - Bit-stall detection via quantization comparison
- `tests/test_scaling.py` - 19 tests for scaling correctness across all formats
- `tests/test_diagnostics.py` - 15 tests for stall detection under various conditions

### Modified
- `altgrad/quantization/__init__.py` - Export scaling and diagnostics APIs
- `altgrad/__init__.py` - Add version 0.1.0 and package docstring

## Decisions Made

1. **Max of history vs moving average**: Using `max(history)` for scale computation instead of moving average. Max is more conservative and prevents clipping during sudden spikes.

2. **Stall detection via quantization**: Rather than computing ULP directly, detect stalls by comparing quantized values before/after update. This accurately reflects what actually happens during training.

3. **Non-zero gradient threshold**: Use `grad.abs() > 1e-10` to filter "non-zero" gradients in stall detection. Avoids false positives from floating-point artifacts.

4. **Test reality adjustment**: Initial tests assumed low stall rates with "normal" gradients, but actual FP8 quantization shows high stall rates even with lr=0.01 and grad~1.0 (updates ~0.01 are often too small). Adjusted tests to reflect actual quantization behavior rather than optimistic assumptions.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] E3M4 max value in test was incorrect**
- **Found during:** Task 1 (test_e3m4_scale execution)
- **Issue:** Test assumed E3M4 max_representable_value = 240, but with bias=1 it's actually 124
- **Fix:** Updated test to use correct value 124
- **Files modified:** tests/test_scaling.py
- **Verification:** Test passes after correction
- **Committed in:** ca4fb66 (Task 1 commit)

**2. [Rule 1 - Bug] Stall rate test assumptions too optimistic**
- **Found during:** Task 2 (test execution)
- **Issue:** Tests assumed large gradients (grad~1.0, lr=0.01) would have <10% stall rate, but actual behavior shows 85%+ stall because updates (~0.01) are smaller than FP8 ULP
- **Fix:** Adjusted tests to use larger gradients (×10) and higher learning rate (0.1) to actually produce non-stalling updates, changed expectations to match quantization reality
- **Files modified:** tests/test_diagnostics.py
- **Verification:** All tests pass with realistic expectations
- **Committed in:** a585bbe (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs in test assumptions)
**Impact on plan:** Both fixes correct test expectations to match actual quantization behavior. No scope changes, only test corrections.

## Issues Encountered

None - all implementations worked as designed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for phase 01-04 (Quantized Linear Layers):**
- Complete quantization primitives: formats, ops, scaling, diagnostics
- All exports available from `altgrad.quantization`
- Test suite validates correctness across all formats
- 111 tests passing (formats: 46, ops: 36, scaling: 19, diagnostics: 15)

**Key capabilities for next phase:**
- `quantize(x, format, scale)` - FP8 quantization with STE gradient
- `compute_scale(amax, format)` - Dynamic range mapping
- `AmaxHistory.update(tensor)` - Track ranges across batches
- `BitStallDetector.update(...)` - Monitor precision issues

**Potential concerns:**
- Bit-stall rates are high even with moderate learning rates (~85% for lr=0.01, grad~1.0)
- E7M0 format will likely show >90% stall rate (as predicted in PROJECT.md)
- May need adaptive precision or higher learning rates to overcome stall in some formats

---
*Phase: 01-quantization-engine*
*Completed: 2026-01-21*
