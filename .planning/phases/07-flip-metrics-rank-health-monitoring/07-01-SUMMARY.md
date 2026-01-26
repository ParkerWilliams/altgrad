---
phase: 07-flip-metrics-rank-health-monitoring
plan: 01
subsystem: quantization
tags: [fp8, diagnostics, weight-flip, training-metrics]

# Dependency graph
requires:
  - phase: 01-quantization-engine
    provides: quantize function, FP8Format class, ops.py patterns
provides:
  - WeightFlipTracker class for tracking weight transitions
  - compute_flip_rate function for flip rate calculation
  - Unit tests for flip metrics
affects:
  - 07-02 (rank health monitoring may use flip metrics)
  - experiments (flip metrics for training diagnostics)
  - analysis (flip data for format comparison)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pre/post step pattern for tracking weight changes"
    - "Per-layer Dict tracking for training diagnostics"

key-files:
  created:
    - altgrad/quantization/flip_metrics.py
    - tests/test_flip_metrics.py
  modified:
    - altgrad/quantization/__init__.py

key-decisions:
  - "Clone quantized tensor in snapshot (not reference) to avoid mutation issues"
  - "Delete snapshot after compute_flips_post_step to prevent memory accumulation"
  - "Cumulative flip counts across steps for epoch-level statistics"

patterns-established:
  - "snapshot_pre_step / compute_post_step lifecycle for training metrics"
  - "get_*_counts() / get_*_rates() accessor pattern for diagnostics"

# Metrics
duration: 5min
completed: 2026-01-25
---

# Phase 7 Plan 01: Weight Flip Metrics Summary

**WeightFlipTracker for counting FP8 representation changes during training with per-layer statistics**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-25T20:20:00Z
- **Completed:** 2026-01-25T20:25:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created flip_metrics.py with WeightFlipTracker class and compute_flip_rate function
- Implemented snapshot_pre_step / compute_flips_post_step lifecycle for tracking weight changes
- Added per-layer flip counts and flip rates for training diagnostics
- Created 13 unit tests covering all functionality
- Exported new classes from altgrad.quantization package

## Task Commits

Each task was committed atomically:

1. **Task 1: Create flip_metrics.py with WeightFlipTracker** - `015225e` (feat)
2. **Task 2: Add flip_metrics exports and unit tests** - `1d31e72` (feat)

## Files Created/Modified
- `altgrad/quantization/flip_metrics.py` - WeightFlipTracker class and compute_flip_rate function
- `tests/test_flip_metrics.py` - 13 unit tests for flip metrics
- `altgrad/quantization/__init__.py` - Added exports for WeightFlipTracker, compute_flip_rate

## Decisions Made
- Clone quantized tensor in snapshot (not reference) to avoid mutation issues during training
- Delete snapshot after compute_flips_post_step to prevent memory accumulation
- Support cumulative flip counts across multiple steps (rate can exceed 1.0 over epochs)
- Follow diagnostics.py pattern for consistency with BitStallDetector

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- WeightFlipTracker ready for integration with training loop
- Can be logged to W&B alongside bit-stall metrics
- Complements bit-stall detection for complete quantization dynamics picture
- Ready for 07-02 rank health monitoring implementation

---
*Phase: 07-flip-metrics-rank-health-monitoring*
*Plan: 01*
*Completed: 2026-01-25*
