---
status: complete
phase: 04-custom-format-testing
source: [04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md]
started: 2026-01-22T01:30:00Z
completed: 2026-01-22T01:45:00Z
---

## Current Test

[all tests complete]

## Tests

### 1. PartitionRelativeClipper Threshold Scaling
expected: E3M4 format gets clip threshold scaled to ~0.2% of base (ratio = 124/57344)
result: pass

### 2. EmergencyMantissaShift Fallback Chain
expected: E7M0 -> E5M2, E1M6 -> E3M4, E0M7 -> E3M4, E5M2 -> None
result: pass

### 3. Stiffness Field E0M7 Constant
expected: E0M7 (fixed-point) returns constant stiffness 1/128 = 0.0078125 for all non-zero weights
result: pass

### 4. ULP Computation Uses torch.nextafter
expected: Unchanged weights (before == after) show ULP distance of 0
result: pass

### 5. FormatExperimentRunner Initialization
expected: Runner initializes with clipper and shifter when config enables them
result: pass

### 6. Format Configs Exist with 500 Steps
expected: 5 format configs (e5m2_short, e0m7, e1m6, e3m4, e7m0) all have max_steps: 500 and seed: 42
result: pass

## Summary

total: 6
passed: 6
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
