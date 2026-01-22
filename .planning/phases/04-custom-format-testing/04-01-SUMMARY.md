---
phase: 04-custom-format-testing
plan: 01
subsystem: quantization
tags: [stability, gradient-clipping, nan-detection, format-fallback, fp8]

# Dependency graph
requires:
  - phase: 01-quantization-engine
    provides: FP8Format with max_representable_value property
  - phase: 02-baseline-validation
    provides: BitStallDetector for stall rate measurement
provides:
  - PartitionRelativeClipper for format-aware gradient clipping
  - EmergencyMantissaShift for runtime format fallback on NaN/stall
affects: [04-02-advanced-diagnostics, 04-03-exotic-format-runner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Format-relative scaling using E5M2 as baseline
    - Consecutive trigger pattern for NaN patience
    - Fallback chain for format degradation

key-files:
  created:
    - altgrad/quantization/stability.py
    - tests/test_stability.py
  modified:
    - altgrad/quantization/__init__.py

key-decisions:
  - "E5M2_MAX=57344 as baseline for threshold scaling"
  - "1% overflow threshold for clipper activation"
  - "3 consecutive NaNs or >50% stall rate triggers shift"
  - "Fallback chain: E7M0->E5M2, E1M6->E3M4, E0M7->E3M4, E3M4->E5M2, E5M2->None"

patterns-established:
  - "Partition-relative scaling: threshold = base * (format_max / baseline_max)"
  - "Consecutive trigger with reset: counter resets on valid batch"

# Metrics
duration: 3min
completed: 2026-01-22
---

# Phase 4 Plan 01: Stability Interventions Summary

**Format-aware gradient clipping (STAB-05) and emergency mantissa shift (STAB-06) for exotic format experiments**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-22T01:07:58Z
- **Completed:** 2026-01-22T01:10:43Z
- **Tasks:** 3 (TDD RED-GREEN)
- **Files modified:** 3

## Accomplishments
- PartitionRelativeClipper scales gradient clip threshold by format dynamic range ratio vs E5M2 baseline
- EmergencyMantissaShift detects training collapse (NaN patience, stall rate) and recommends format fallback
- Fallback chain enables graceful degradation: E7M0->E5M2, E1M6->E3M4, E0M7->E3M4
- 18 comprehensive tests covering all success criteria

## Task Commits

Each task was committed atomically:

1. **Task 1: RED - Write failing tests** - `964b5c2` (test)
2. **Task 2: GREEN - Implement PartitionRelativeClipper** - `5af42f7` (feat)
3. **Task 3: GREEN - EmergencyMantissaShift** - bundled with Task 2 (same file)

_Note: Task 3 implementation was bundled with Task 2 since both classes are in stability.py_

## Files Created/Modified

- `altgrad/quantization/stability.py` - PartitionRelativeClipper and EmergencyMantissaShift classes (162 lines)
- `tests/test_stability.py` - Comprehensive TDD tests for stability interventions (250 lines)
- `altgrad/quantization/__init__.py` - Export stability classes

## Decisions Made

- **E5M2_MAX = 57344 as baseline:** Standard FP8 format, reference point for scaling
- **1% overflow threshold:** Clip only when overflow_rate >= 0.01 (not always-on)
- **3 consecutive NaNs or >50% stall:** Conservative triggers to avoid false positives
- **Fallback to higher-range formats:** E7M0/E1M6/E0M7 fall back to formats with better dynamic range

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - implementation straightforward following existing quantization patterns.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Stability interventions ready for integration with exotic format runner (04-03)
- PartitionRelativeClipper can be used during training to prevent overflow
- EmergencyMantissaShift monitors for collapse and recommends format changes
- Fallback chain enables automatic degradation path for unstable formats

---
*Phase: 04-custom-format-testing*
*Completed: 2026-01-22*
