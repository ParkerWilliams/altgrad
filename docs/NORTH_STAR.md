# North Star: Analysis Traceability Matrix

**Purpose:** This document defines the complete set of metrics that must be computed and reported for every datatype experiment. Every cell traces to its computation source and report destination.

**Research Question:** Which FP8/FP16 datatype benefits most from manifold-aware gradient updates?

---

## Datatypes Under Test

### 8-bit Formats (E+M=7)
| Format | E | M | Description |
|--------|---|---|-------------|
| FP8_E0M7 | 0 | 7 | Fixed-point, max precision |
| FP8_E1M6 | 1 | 6 | Minimal range |
| FP8_E2M5 | 2 | 5 | Low range |
| FP8_E3M4 | 3 | 4 | Moderate balance |
| FP8_E4M3 | 4 | 3 | Common ML format |
| FP8_E5M2 | 5 | 2 | Standard (like FP16 scaled) |
| FP8_E6M1 | 6 | 1 | Wide range |
| FP8_E7M0 | 7 | 0 | Maximum range, no precision |

### 16-bit Formats (E+M=15)
| Format | E | M | Description |
|--------|---|---|-------------|
| FP16 | 5 | 10 | IEEE half precision |
| BF16 | 8 | 7 | Brain float (wide range) |
| FP16_E0M15 | 0 | 15 | Fixed-point 16-bit |
| FP16_E4M11 | 4 | 11 | Similar to FP8_E4M3 range |
| FP16_E7M8 | 7 | 8 | Similar to FP8_E7M0 range |

---

## Traceability Matrix

Every metric has two lines:
- **Line 1:** Experiment code (where metric is computed)
- **Line 2:** Report code (where metric appears in output)

All paths relative to `altgrad/`

### THROUGHPUT

| Metric | Experiment | Report |
|--------|------------|--------|
| samples/sec | `training/classification_metrics.py:391` | `analysis/results_table.py:241` |
| tokens/sec | `training/classification_metrics.py:392` | `analysis/results_table.py:241` |

### CLASSIFIER PERFORMANCE

| Metric | Experiment | Report |
|--------|------------|--------|
| F1 micro | `training/classification_metrics.py:178` | `analysis/results_table.py:259` |
| F1 macro | `training/classification_metrics.py:183` | `analysis/results_table.py:259` |
| Precision micro | `training/classification_metrics.py:179` | `analysis/results_table.py:259` |
| Recall micro | `training/classification_metrics.py:180` | `analysis/results_table.py:259` |
| Subset accuracy | `training/classification_metrics.py:210` | `analysis/results_table.py:259` |
| Hamming loss | `training/classification_metrics.py:206` | `analysis/results_table.py:41` (CSV/JSON) |
| ROC-AUC micro | `training/classification_metrics.py:337` | `analysis/results_table.py:42` (CSV/JSON) |
| PR-AUC micro | `training/classification_metrics.py:368` | CSV/JSON export |

### RANK HEALTH

| Metric | Experiment | Report |
|--------|------------|--------|
| Stable rank (init/final) | `quantization/rank_health.py:44-87` | `analysis/results_table.py:280` |
| Effective rank (init/final) | `quantization/rank_health.py:90-143` | `analysis/results_table.py:280` |
| Collapse detection | `quantization/rank_health.py:366-397` | `analysis/results_table.py:96-109` |

### TRAINING EFFICIENCY

| Metric | Experiment | Report |
|--------|------------|--------|
| Total time (sec) | `training/classification_trainer.py:396` | `analysis/results_table.py:314` |
| Total flips | `quantization/flip_metrics.py:177-225` | `analysis/results_table.py:307` |
| Stall ratio | `quantization/flip_metrics.py:278-301` | `analysis/results_table.py:309` |

### DERIVED METRICS

| Metric | Computation | Report |
|--------|-------------|--------|
| F1 delta (Manifold - AdamW) | `analysis/results_table.py:86-88` | `analysis/results_table.py:261` |
| Stall delta | `analysis/results_table.py:91-93` | `analysis/results_table.py:311` |

---

## Experimental Design

### Independent Variables
1. **Datatype:** All formats listed above
2. **Optimizer:** AdamW vs ManifoldAdamW

### Dependent Variables
All metrics in the traceability matrix above.

### Experimental Matrix
- 8 FP8 formats × 2 optimizers = 16 conditions
- 5+ FP16 formats × 2 optimizers = 10+ conditions
- **Total:** 26+ experimental conditions

---

## Success Criteria

The experiment answers: **"Which format benefits most from ManifoldAdamW?"**

Measured by:
1. **F1 Delta:** `F1_micro(Manifold) - F1_micro(AdamW)` per format
2. **Rank Preservation:** Did ManifoldAdamW prevent collapse where AdamW failed?
3. **Stall Reduction:** Did ManifoldAdamW reduce gradient stalls?

### Expected Outcomes

| Format Type | Hypothesis |
|-------------|------------|
| Low mantissa (E5M2, E6M1, E7M0) | Highest benefit from ManifoldAdamW (coarse grids) |
| High mantissa (E0M7, E1M6) | Lower benefit (finer grids, less stalling) |
| 16-bit formats | Minimal benefit (already precise enough) |

---

## File Reference

| File | Purpose |
|------|---------|
| `training/classification_metrics.py` | Metric computation during eval |
| `training/classification_trainer.py` | Training loop with rank/flip integration |
| `quantization/rank_health.py` | Stable/effective rank, collapse detection |
| `quantization/flip_metrics.py` | Weight flip counting, stall ratios |
| `analysis/results_table.py` | Report generation (plaintext, CSV, JSON) |

---

## Non-Negotiables

1. **Every datatype must be tested with both optimizers**
2. **Every metric in this matrix must be computed and reported**
3. **Rank collapse monitoring must run during training** (not just post-hoc)
4. **Results table must show per-format F1 delta** (the core research answer)

---

*This document is the authoritative reference for altgrad experiment scope.*
