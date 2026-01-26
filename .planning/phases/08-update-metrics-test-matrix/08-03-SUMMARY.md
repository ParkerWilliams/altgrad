---
phase: 08-update-metrics-test-matrix
plan: 03
subsystem: training
tags: [rank-health, monitoring, thresholds, classifier, lm_head, c_proj]

# Dependency graph
requires:
  - phase: 07-flip-metrics-rank-health-monitoring
    provides: RankHealthMonitor base implementation
provides:
  - Configurable critical_threshold_multiplier for classifier layers
  - get_threshold_for_layer() public method for per-layer thresholds
  - Pattern-based critical layer detection (lm_head, c_proj)
affects: [experiments, training, rank-monitoring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern matching: any(pattern in name for pattern in self.critical_layers)"
    - "Configurable multiplier for stricter thresholds on critical layers"

key-files:
  created: []
  modified:
    - altgrad/quantization/rank_health.py
    - tests/test_rank_health.py

key-decisions:
  - "Default critical_layers = ['lm_head', 'c_proj'] for classifier/projection layers"
  - "Default multiplier 0.5 gives classifiers 15% threshold vs 30% for other layers"
  - "Pattern-based matching allows flexible layer name patterns"

patterns-established:
  - "Classifier-specific monitoring: stricter thresholds for output-critical layers"
  - "Configurable multiplier pattern for tunable threshold strictness"

# Metrics
duration: 1min
completed: 2026-01-26
---

# Phase 8 Plan 3: Classifier-Specific Thresholds Summary

**Configurable critical_threshold_multiplier (default 0.5) gives classifier layers (lm_head, c_proj) stricter 15% rank collapse warning vs 30% for other layers**

## Performance

- **Duration:** 1 min
- **Started:** 2026-01-26T20:05:31Z
- **Completed:** 2026-01-26T20:06:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- critical_threshold_multiplier parameter added to RankHealthMonitor.__init__ (default 0.5)
- get_threshold_for_layer() public method returns per-layer threshold based on pattern matching
- check_warnings() now uses configurable per-layer thresholds
- 5 new tests covering all threshold behavior scenarios
- Default behavior: lm_head/c_proj warn at 15% drop, other layers at 30%

## Task Commits

Each task was committed atomically:

1. **Task 1: Add critical_threshold_multiplier to RankHealthMonitor** - `9577b37` (feat)
2. **Task 2: Add tests for classifier-specific thresholds** - `42e15e4` (test)

## Files Created/Modified
- `altgrad/quantization/rank_health.py` - Added critical_threshold_multiplier parameter and get_threshold_for_layer() method
- `tests/test_rank_health.py` - Added TestClassifierSpecificThresholds class with 5 tests

## Decisions Made
- **Default critical_layers:** ["lm_head", "c_proj"] - these are output-critical classifier/projection layers
- **Default multiplier:** 0.5 gives classifiers half the threshold (0.3 * 0.5 = 0.15)
- **Pattern matching:** any(pattern in name for pattern in self.critical_layers) for flexible matching

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- RankHealthMonitor complete with configurable classifier-specific thresholds
- Ready for use in format comparison experiments
- Integrates with existing rank health monitoring infrastructure

---
*Phase: 08-update-metrics-test-matrix*
*Completed: 2026-01-26*
