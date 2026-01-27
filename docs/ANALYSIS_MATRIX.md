# Analysis Matrix: Test → Metric → Report Traceability

Every cell in the results table traced to its test and report location.

---

## THROUGHPUT

| Metric | Test | Report |
|--------|------|--------|
| **samples_per_sec** | | |
| Test | `altgrad/training/classification_metrics.py:286` `compute_throughput_metrics()` |
| Report | `altgrad/analysis/results_table.py:162` `row.throughput.samples_per_sec` |
| **tokens_per_sec** | | |
| Test | `altgrad/training/classification_metrics.py:287` `compute_throughput_metrics()` |
| Report | `altgrad/analysis/results_table.py:162` `row.throughput.tokens_per_sec` |

---

## CLASSIFIER PERFORMANCE - AdamW

| Metric | Test | Report |
|--------|------|--------|
| **F1 micro** | | |
| Test | `altgrad/training/classification_metrics.py:178` `metrics.f1_micro = (2 * micro_precision * micro_recall / ...)` |
| Report | `altgrad/analysis/results_table.py:176` `a.f1_micro` |
| **F1 macro** | | |
| Test | `altgrad/training/classification_metrics.py:183` `metrics.f1_macro = label_f1.mean()` |
| Report | `altgrad/analysis/results_table.py:176` `a.f1_macro` |
| **Precision micro** | | |
| Test | `altgrad/training/classification_metrics.py:179` `metrics.precision_micro = micro_precision` |
| Report | `altgrad/analysis/results_table.py:176` `a.precision_micro` |
| **Recall micro** | | |
| Test | `altgrad/training/classification_metrics.py:180` `metrics.recall_micro = micro_recall` |
| Report | `altgrad/analysis/results_table.py:176` `a.recall_micro` |
| **Subset accuracy** | | |
| Test | `altgrad/training/classification_metrics.py:210` `metrics.subset_accuracy = exact_match.mean()` |
| Report | `altgrad/analysis/results_table.py:176` `a.subset_accuracy` |
| **Hamming loss** | | |
| Test | `altgrad/training/classification_metrics.py:206` `metrics.hamming_loss = ((preds != labels).float().sum() / ...)` |
| Report | `altgrad/analysis/results_table.py` (not in plaintext, in CSV/JSON) |
| **ROC-AUC micro** | | |
| Test | `altgrad/training/classification_metrics.py:223-247` `_compute_roc_auc()` |
| Report | `altgrad/analysis/results_table.py` (not in plaintext, in CSV/JSON) |
| **PR-AUC micro** | | |
| Test | `altgrad/training/classification_metrics.py:249-273` `_compute_pr_auc()` |
| Report | `altgrad/analysis/results_table.py` (not in plaintext, in CSV/JSON) |

---

## CLASSIFIER PERFORMANCE - ManifoldAdamW

| Metric | Test | Report |
|--------|------|--------|
| **F1 micro** | | |
| Test | `altgrad/training/classification_metrics.py:178` (same computation, different optimizer) |
| Report | `altgrad/analysis/results_table.py:177` `m.f1_micro` |
| **F1 macro** | | |
| Test | `altgrad/training/classification_metrics.py:183` |
| Report | `altgrad/analysis/results_table.py:177` `m.f1_macro` |
| **Precision micro** | | |
| Test | `altgrad/training/classification_metrics.py:179` |
| Report | `altgrad/analysis/results_table.py:177` `m.precision_micro` |
| **Recall micro** | | |
| Test | `altgrad/training/classification_metrics.py:180` |
| Report | `altgrad/analysis/results_table.py:177` `m.recall_micro` |
| **Subset accuracy** | | |
| Test | `altgrad/training/classification_metrics.py:210` |
| Report | `altgrad/analysis/results_table.py:177` `m.subset_accuracy` |

---

## CLASSIFIER PERFORMANCE - Derived

| Metric | Test | Report |
|--------|------|--------|
| **Δ F1 (Manifold - AdamW)** | | |
| Test | `altgrad/analysis/results_table.py:79` `return self.classifier_manifold.f1_micro - self.classifier_adamw.f1_micro` |
| Report | `altgrad/analysis/results_table.py:178` `delta_str` |

---

## RANK HEALTH - AdamW

| Metric | Test | Report |
|--------|------|--------|
| **Stable rank init** | | |
| Test | `altgrad/quantization/rank_health.py:44-87` `compute_stable_rank()` |
| Report | `altgrad/analysis/results_table.py:196` `ra.stable_rank_init` |
| **Stable rank final** | | |
| Test | `altgrad/quantization/rank_health.py:44-87` `compute_stable_rank()` (at end of training) |
| Report | `altgrad/analysis/results_table.py:196` `ra.stable_rank_final` |
| **Effective rank init** | | |
| Test | `altgrad/quantization/rank_health.py:90-143` `compute_effective_rank()` |
| Report | `altgrad/analysis/results_table.py:196` `ra.effective_rank_init` |
| **Effective rank final** | | |
| Test | `altgrad/quantization/rank_health.py:90-143` `compute_effective_rank()` (at end of training) |
| Report | `altgrad/analysis/results_table.py:196` `ra.effective_rank_final` |
| **Collapse detected** | | |
| Test | `altgrad/quantization/rank_health.py:366-397` `RankHealthMonitor.check_warnings()` |
| Report | `altgrad/analysis/results_table.py:85-89` `rank_collapse_adamw` property |

---

## RANK HEALTH - ManifoldAdamW

| Metric | Test | Report |
|--------|------|--------|
| **Stable rank init** | | |
| Test | `altgrad/quantization/rank_health.py:44-87` `compute_stable_rank()` |
| Report | `altgrad/analysis/results_table.py:197` `rm.stable_rank_init` |
| **Stable rank final** | | |
| Test | `altgrad/quantization/rank_health.py:44-87` `compute_stable_rank()` |
| Report | `altgrad/analysis/results_table.py:197` `rm.stable_rank_final` |
| **Collapse detected** | | |
| Test | `altgrad/quantization/rank_health.py:366-397` `RankHealthMonitor.check_warnings()` |
| Report | `altgrad/analysis/results_table.py:91-95` `rank_collapse_manifold` property |

---

## TRAINING EFFICIENCY - AdamW

| Metric | Test | Report |
|--------|------|--------|
| **Total time (sec)** | | |
| Test | `altgrad/training/classification_trainer.py:389` `step_time = time.time() - step_start` (accumulated) |
| Report | `altgrad/analysis/results_table.py:220` `ta.total_time_sec` |
| **Total flips** | | |
| Test | `altgrad/quantization/flip_metrics.py:177-225` `WeightFlipTracker.compute_flips_post_step()` |
| Report | `altgrad/analysis/results_table.py:226` `ta.total_flips` |
| **Mean stall ratio** | | |
| Test | `altgrad/quantization/flip_metrics.py:278-301` `WeightFlipTracker.get_stall_ratios()` |
| Report | `altgrad/analysis/results_table.py:227` `ta.mean_stall_ratio` |

---

## TRAINING EFFICIENCY - ManifoldAdamW

| Metric | Test | Report |
|--------|------|--------|
| **Total time (sec)** | | |
| Test | `altgrad/training/classification_trainer.py:389` |
| Report | `altgrad/analysis/results_table.py:221` `tm.total_time_sec` |
| **Total flips** | | |
| Test | `altgrad/quantization/flip_metrics.py:177-225` |
| Report | `altgrad/analysis/results_table.py:226` `tm.total_flips` |
| **Mean stall ratio** | | |
| Test | `altgrad/quantization/flip_metrics.py:278-301` |
| Report | `altgrad/analysis/results_table.py:227` `tm.mean_stall_ratio` |

---

## TRAINING EFFICIENCY - Derived

| Metric | Test | Report |
|--------|------|--------|
| **Δ Stall (Manifold - AdamW)** | | |
| Test | `altgrad/analysis/results_table.py:83` `return self.training_manifold.mean_stall_ratio - self.training_adamw.mean_stall_ratio` |
| Report | `altgrad/analysis/results_table.py:231` `delta_stall` |

---

## SUMMARY METRICS

| Metric | Test | Report |
|--------|------|--------|
| **Best 8-bit format** | | |
| Test | `altgrad/analysis/results_table.py:241-248` loop finding max `f1_delta` where `bits == 8` |
| Report | `altgrad/analysis/results_table.py:269` `<-- BEST 8-bit` marker |
| **Best 16-bit format** | | |
| Test | `altgrad/analysis/results_table.py:241-248` loop finding max `f1_delta` where `bits == 16` |
| Report | `altgrad/analysis/results_table.py:271` `<-- BEST 16-bit` marker |

---

## INTEGRATION POINTS

### Where metrics are computed during training:

| Stage | File:Line | Metrics Computed |
|-------|-----------|------------------|
| Train step | `classification_trainer.py:302-335` | loss, grad_stats |
| Post optimizer | `classification_trainer.py:334` | flip_metrics |
| Rank interval | `classification_trainer.py:338-344` | stable_rank, effective_rank, collapse_warnings |
| Eval | `classification_trainer.py:359-371` | all ClassificationMetrics |
| Throughput | `classification_trainer.py:389-394` | samples_per_sec, tokens_per_sec |

### Where metrics are logged:

| Destination | File:Line | What |
|-------------|-----------|------|
| Console | `classification_trainer.py:432-455` | loss, samples/s, stable_rank, stall_ratio, flips |
| W&B | `classification_trainer.py:458` | all train/* metrics |
| W&B eval | `classification_trainer.py:476` | all eval/* metrics |
| Checkpoint | `classification_trainer.py:483-492` | model, optimizer, best_f1, config |

### Where results table is populated:

| Source | File:Line | Populates |
|--------|-----------|-----------|
| Experiment runner | `run_classification.py:98-106` | classifier metrics from `trainer.evaluate()` |
| Results table | `results_table.py:119-136` | `add_result()` method |

---

## FORMAT SPECIFICATIONS

| Format | Bits | E | M | Bias | Max Value | Min Positive | File:Line |
|--------|------|---|---|------|-----------|--------------|-----------|
| FP8_E0M7 | 8 | 0 | 7 | 0 | 0.992 | 0.0078 | `formats.py:299` |
| FP8_E1M6 | 8 | 1 | 6 | 0 | 3.97 | 0.031 | `formats.py:300` |
| FP8_E2M5 | 8 | 2 | 5 | 1 | 7.88 | 0.031 | `formats.py:301` |
| FP8_E3M4 | 8 | 3 | 4 | 3 | 31 | 0.016 | `formats.py:302` |
| FP8_E4M3 | 8 | 4 | 3 | 7 | 480 | 0.002 | `formats.py:303` |
| FP8_E5M2 | 8 | 5 | 2 | 15 | 57344 | 1.5e-5 | `formats.py:304` |
| FP8_E6M1 | 8 | 6 | 1 | 31 | 6.4e9 | 4.7e-10 | `formats.py:305` |
| FP8_E7M0 | 8 | 7 | 0 | 63 | 1.8e19 | 2.2e-19 | `formats.py:306` |
| FP16 | 16 | 5 | 10 | 15 | 65504 | 6.0e-8 | `formats.py:314` |
| BF16 | 16 | 8 | 7 | 127 | 3.4e38 | 9.2e-41 | `formats.py:315` |
| FP16_E0M15 | 16 | 0 | 15 | 0 | 1.0 | 3.1e-5 | `formats.py:318` |
| FP16_E1M14 | 16 | 1 | 14 | 0 | 4.0 | 1.2e-4 | `formats.py:319` |
| FP16_E2M13 | 16 | 2 | 13 | 1 | 8.0 | 1.2e-4 | `formats.py:320` |
| FP16_E3M12 | 16 | 3 | 12 | 3 | 32 | 6.1e-5 | `formats.py:321` |
| FP16_E4M11 | 16 | 4 | 11 | 7 | 512 | 7.6e-6 | `formats.py:322` |
| FP16_E6M9 | 16 | 6 | 9 | 31 | 8.6e9 | 1.8e-12 | `formats.py:324` |
| FP16_E7M8 | 16 | 7 | 8 | 63 | 3.7e19 | 8.5e-22 | `formats.py:325` |
| FP16_E9M6 | 16 | 9 | 6 | 255 | 2.3e77 | 5.4e-79 | `formats.py:327` |
| FP16_E10M5 | 16 | 10 | 5 | 511 | 2.6e154 | 9.3e-156 | `formats.py:328` |
| FP16_E11M4 | 16 | 11 | 4 | 1023 | inf | 1.4e-309 | `formats.py:329` |
| FP16_E12M3 | 16 | 12 | 3 | 2047 | inf | 0 | `formats.py:330` |
| FP16_E13M2 | 16 | 13 | 2 | 4095 | inf | 0 | `formats.py:331` |
| FP16_E14M1 | 16 | 14 | 1 | 8191 | inf | 0 | `formats.py:332` |
| FP16_E15M0 | 16 | 15 | 0 | 16383 | inf | 0 | `formats.py:333` |

---

## CROSS-FORMAT COMPARISONS (Same E, Different Bits)

| E bits | 8-bit Format | 16-bit Format | Comparison |
|--------|--------------|---------------|------------|
| E=0 | FP8_E0M7 | FP16_E0M15 | Fixed-point: 7 vs 15 mantissa bits |
| E=1 | FP8_E1M6 | FP16_E1M14 | Minimal range: 6 vs 14 mantissa bits |
| E=2 | FP8_E2M5 | FP16_E2M13 | Low range: 5 vs 13 mantissa bits |
| E=3 | FP8_E3M4 | FP16_E3M12 | Moderate: 4 vs 12 mantissa bits |
| E=4 | FP8_E4M3 | FP16_E4M11 | Common ML: 3 vs 11 mantissa bits |
| E=5 | FP8_E5M2 | FP16 (E5M10) | Standard: 2 vs 10 mantissa bits |
| E=6 | FP8_E6M1 | FP16_E6M9 | Wide range: 1 vs 9 mantissa bits |
| E=7 | FP8_E7M0 | FP16_E7M8 | Very wide: 0 vs 8 mantissa bits |
| E=8 | N/A | BF16 (E8M7) | BF16 only |

---

*Generated for altgrad experiment traceability*
