---
phase: 07-flip-metrics-rank-health-monitoring
plan: 02
subsystem: diagnostics
tags: [rank, svd, matrix-health, ema, collapse-detection]

# Dependency graph
requires:
  - phase: 07-01
    provides: flip metrics module pattern
provides:
  - compute_stable_rank for ||W||_F^2 / ||W||_2^2
  - compute_effective_rank for exp(entropy) measure
  - RankTrendDetector for EMA-based collapse warning
  - RankHealthMonitor for per-layer monitoring
  - compute_rank_stats for training metrics integration
affects: [training-monitoring, experiment-analysis]

# Tech tracking
tech-stack:
  added: [torch.linalg.svdvals]
  patterns: [EMA trend detection with warmup, per-layer monitoring]

key-files:
  created:
    - altgrad/quantization/rank_health.py
    - tests/test_rank_health.py
  modified:
    - altgrad/quantization/__init__.py
    - altgrad/training/metrics.py

key-decisions:
  - "Use torch.linalg.svdvals for efficient SVD (10-50x faster than full SVD)"
  - "Reshape conv weights to 2D (out_features, -1) before SVD"
  - "EMA warmup window before trend detection to avoid initialization noise"
  - "Per-layer tracking with stricter thresholds for critical layers (lm_head, c_proj)"

patterns-established:
  - "Rank metrics pattern: stable_rank + effective_rank as complementary measures"
  - "Trend detector pattern: EMA with warmup, threshold_pct for drop warning"

# Metrics
duration: 6min
completed: 2026-01-25
---

# Phase 7 Plan 02: Rank Health Monitoring Summary

**Stable rank and effective rank computation with EMA-based collapse early warning using torch.linalg.svdvals**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-25T15:20:00Z
- **Completed:** 2026-01-25T15:26:00Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Implemented compute_stable_rank: ||W||_F^2 / ||W||_2^2 for effective dimension measurement
- Implemented compute_effective_rank: exp(entropy) of normalized singular values
- Created RankTrendDetector with EMA warmup and threshold-based collapse warning
- Created RankHealthMonitor for per-layer rank tracking with spectral norm
- Added compute_rank_stats to training/metrics.py for aggregate rank statistics
- 23 comprehensive unit tests all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create rank_health.py with rank computation functions** - `7be9504` (feat)
2. **Task 2: Add compute_rank_stats to metrics.py and exports** - `dd9161b` (feat)
3. **Task 3: Create unit tests for rank health** - `b9059f3` (test)

## Files Created/Modified
- `altgrad/quantization/rank_health.py` - Core rank computation functions and monitoring classes
- `altgrad/quantization/__init__.py` - Export rank health functions
- `altgrad/training/metrics.py` - compute_rank_stats for training integration
- `tests/test_rank_health.py` - 23 unit tests covering all functions

## Decisions Made
- **torch.linalg.svdvals:** Used instead of full SVD for 10-50x speedup (only need singular values)
- **2D reshape:** Reshape higher-dim weights (conv) to (out_features, -1) before SVD
- **EMA warmup:** Wait `window` steps before trend detection to avoid initialization noise
- **Threshold-based warning:** Warn when EMA drops by `threshold_pct` from initial baseline
- **Critical layer stricter threshold:** lm_head/c_proj get 0.5x the normal threshold

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed without issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Rank health monitoring ready for integration with training loop
- Can be combined with flip metrics for comprehensive quantization health dashboard
- Per-layer rank tracking enables identification of problematic layers during training

---
*Phase: 07-flip-metrics-rank-health-monitoring*
*Completed: 2026-01-25*
