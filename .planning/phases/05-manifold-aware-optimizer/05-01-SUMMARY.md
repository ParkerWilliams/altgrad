---
phase: 05-manifold-aware-optimizer
plan: 01
subsystem: training
tags: [optimizer, adamw, stiffness, manifold, ulp, fp8, quantization]

# Dependency graph
requires:
  - phase: 04-custom-format-testing
    provides: compute_stiffness_field in advanced_diagnostics.py
provides:
  - ManifoldAdamW optimizer class
  - Stiffness-preconditioned gradient updates
  - Bit-position tracking for ULP movement
affects: [05-02, 05-03, experiment-configs, trainer-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Custom PyTorch optimizer with per-parameter state
    - Stiffness preconditioning before momentum updates
    - Latent integer tracking via bit-position accumulation

key-files:
  created:
    - altgrad/training/optimizer.py
    - tests/test_optimizer.py
  modified:
    - altgrad/training/__init__.py

key-decisions:
  - "Stiffness preconditioning applied BEFORE momentum (precondition gradient, then exp_avg update)"
  - "NaN stiffness (zero weights) replaced with 1.0 neutral multiplier"
  - "Stiffness clamped to max_stiffness (default 1e6) to prevent explosion"
  - "Bit-position tracks signed ULP movement (direction-aware accumulation)"

patterns-established:
  - "Custom optimizer pattern: subclass torch.optim.Optimizer, lazy state init, @torch.no_grad() step"
  - "Stiffness preconditioning: multiply gradient by compute_stiffness_field before Adam moments"
  - "ULP tracking: delta_ulps = (after - before) / safe_ulp, accumulate in state"

# Metrics
duration: 8min
completed: 2026-01-22
---

# Phase 5 Plan 1: ManifoldAdamW Optimizer Summary

**AdamW optimizer with stiffness-preconditioned gradient updates, manifold-aware toggle, and bit-position tracking for FP8 geometry-aware training**

## Performance

- **Duration:** 8 min
- **Started:** 2026-01-23T03:30:00Z
- **Completed:** 2026-01-23T03:38:00Z
- **Tasks:** 3 (TDD: RED, GREEN, REFACTOR)
- **Files modified:** 3

## Accomplishments
- ManifoldAdamW optimizer with stiffness preconditioning (MANI-02)
- Standard vs manifold-aware mode toggle (MANI-03)
- Bit-position tracking for cumulative ULP movement (MANI-04)
- Zero weight and large magnitude edge case handling
- 15 comprehensive tests covering all requirements

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Write failing tests** - `50adfef` (test)
2. **Task 2 (GREEN): Implement ManifoldAdamW** - `7ff9fcd` (feat)
3. **Task 3 (REFACTOR): Clean up** - no changes needed, code was already clean

## Files Created/Modified
- `altgrad/training/optimizer.py` - ManifoldAdamW optimizer (197 lines)
- `tests/test_optimizer.py` - Comprehensive test suite (251 lines, 15 tests)
- `altgrad/training/__init__.py` - Export ManifoldAdamW

## Decisions Made
- Stiffness preconditioning applied BEFORE momentum updates (cleaner, grad becomes precond_grad before exp_avg update)
- NaN stiffness from zero weights replaced with 1.0 (neutral multiplier, no effect on gradient)
- max_stiffness default 1e6 (configurable, prevents explosion at large magnitudes)
- Bit-position tracks signed ULP movement (allows direction analysis in addition to magnitude)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Test bug in `test_weight_decay_decoupled`: used `torch.ones() * 2.0` which creates non-leaf tensor. Fixed by using `torch.full()` instead. This was a test authoring issue, not implementation issue.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- ManifoldAdamW ready for integration with Trainer
- Plan 05-02 can add manifold config options to TrainConfig
- Plan 05-03 can update Trainer to use ManifoldAdamW when configured
- Experiment configs can specify `manifold_aware=True` with format-appropriate `mantissa_bits`

---
*Phase: 05-manifold-aware-optimizer*
*Completed: 2026-01-22*
