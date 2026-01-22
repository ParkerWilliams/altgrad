---
phase: 04-custom-format-testing
plan: 02
subsystem: quantization
tags: [diagnostics, stiffness, ulp, grid-alignment, pearson-correlation]

# Dependency graph
requires:
  - phase: 01-quantization-engine
    provides: quantize function, FP8Format
provides:
  - compute_stiffness_field for format precision analysis
  - grid_alignment_error and grid_alignment_statistics for quantization distance metrics
  - compute_ulp_distance and ulp_statistics for IEEE 754 bit-movement tracking
  - gradient_stiffness_correlation for training dynamics analysis
affects: [04-03, 05-experiment-execution]

# Tech tracking
tech-stack:
  added: []
  patterns: [torch.nextafter for ULP computation, Pearson correlation for diagnostics]

key-files:
  created:
    - altgrad/quantization/advanced_diagnostics.py
    - tests/test_advanced_diagnostics.py
  modified:
    - altgrad/quantization/__init__.py

key-decisions:
  - "E0M7 constant 1/128 stiffness (uniform grid spacing)"
  - "torch.nextafter for IEEE 754 compliant ULP computation"
  - "NaN stiffness for zero weights (undefined)"

patterns-established:
  - "Stiffness formula S = 2^(floor(log2|w|) - M) for precision analysis"
  - "grad_below_stiffness_frac indicates bit-stall risk"

# Metrics
duration: 5min
completed: 2026-01-22
---

# Phase 4 Plan 2: Advanced Diagnostics Summary

**Stiffness field, grid alignment, ULP statistics, and gradient-stiffness correlation for deep quantization analysis**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-22T01:08:05Z
- **Completed:** 2026-01-22T01:13:00Z
- **Tasks:** 3 (TDD pattern: RED, GREEN x2)
- **Files modified:** 3

## Accomplishments
- DIAG-01: Stiffness field computation with E0M7 fixed-point special case
- DIAG-02: Grid alignment using quantize() for exact error measurement
- DIAG-03: Gradient-stiffness Pearson correlation for training dynamics
- DIAG-04: ULP statistics using torch.nextafter for IEEE 754 compliance
- 23 comprehensive tests covering all four diagnostics

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests** - `af67080` (test)
2. **Task 2 (GREEN): Stiffness and grid alignment** - `6104bf3` (feat)
3. **Task 3 (GREEN): ULP and correlation** - `3534422` (feat)

## Files Created/Modified
- `altgrad/quantization/advanced_diagnostics.py` - DIAG-01 to DIAG-04 implementations
- `tests/test_advanced_diagnostics.py` - 23 tests (334 lines)
- `altgrad/quantization/__init__.py` - Added exports for all 6 new functions

## Decisions Made
- E0M7 (M=7) returns constant 1/128 stiffness since fixed-point has uniform grid spacing
- Zero weights return NaN stiffness (log2(0) is undefined)
- torch.nextafter used for ULP computation (IEEE 754 standard, not manual calculation)
- Pearson correlation (not Spearman) for gradient-stiffness relationship
- Valid mask excludes zero weights from correlation calculation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test for clamped values**
- **Found during:** Task 2 (grid alignment tests)
- **Issue:** Test expected E5M2 to clamp overflow to finite max, but E5M2 has has_inf=True so overflows to inf
- **Fix:** Changed test to use E3M4 (no inf support) which actually clamps
- **Files modified:** tests/test_advanced_diagnostics.py
- **Verification:** All grid alignment tests pass
- **Committed in:** 6104bf3 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test)
**Impact on plan:** Test correction aligned with actual format behavior. No scope creep.

## Issues Encountered
None - implementation followed plan specifications.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 6 diagnostic functions exported from altgrad.quantization
- Integration tests verify stiffness predicts stall behavior
- Ready for use in experiment analysis pipeline

---
*Phase: 04-custom-format-testing*
*Completed: 2026-01-22*
