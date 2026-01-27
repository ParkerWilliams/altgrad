# Altgrad Remediation Plan

**Date:** 2026-01-26
**Status:** Active
**Priority:** Critical

## Research Objective

Determine which FP8 datatype benefits most from manifold-aware gradient updates, measured via:
1. Throughput (samples/sec)
2. Classification performance (F1 micro, F1 macro on EUR-Lex)
3. Rank stability (stable rank, effective rank, collapse detection)
4. Training efficiency (weight updates, stall ratio, flips)

## Experimental Design

**Independent Variables:**
- FP8 Format: BF16 (baseline), E5M2, E3M4, E1M6, E0M7, E7M0
- Optimizer: Standard AdamW, ManifoldAdamW

**Dependent Variables:**
- Throughput (samples/sec)
- F1 micro, F1 macro
- Stable rank per layer
- Effective rank per layer
- Stall ratio per layer
- Total flips

**Experimental Matrix:** 6 formats × 2 optimizers = 12 conditions

## Current State Assessment

### Components to PRESERVE (working correctly)

| Component | Location | Status |
|-----------|----------|--------|
| FP8 Format Registry | `altgrad/quantization/formats.py` | Correct |
| Quantize/Dequantize with STE | `altgrad/quantization/ops.py` | Correct |
| Per-tensor scaling | `altgrad/quantization/scaling.py` | Correct |
| ManifoldAdamW | `altgrad/training/optimizer.py` | Correct |
| GridOptim | `altgrad/training/optimizer.py` | Correct |
| Rank health monitoring | `altgrad/quantization/rank_health.py` | Correct |
| Flip/stall metrics | `altgrad/quantization/flip_metrics.py` | Correct |
| Stiffness computation | `altgrad/quantization/advanced_diagnostics.py` | Correct |
| W&B integration | `altgrad/training/callbacks.py` | Correct |
| Checkpoint management | `altgrad/training/checkpoint.py` | Correct |

### Components to REPLACE (fundamentally wrong)

| Component | Current | Required |
|-----------|---------|----------|
| Model | GPT decoder for LM | Encoder + classification head |
| Task | Next-token prediction | Multi-label document classification |
| Loss | Cross-entropy (LM) | Binary cross-entropy (multi-label) |
| Metrics | Perplexity | F1 micro, F1 macro, precision, recall |
| Data loading | Broken/synthetic | EUR-Lex with labels |

## Remediation Tasks

### Phase 1: Data Pipeline

**1.1 Data Loader**
- Load `NLP-AUEB/eurlex` with `trust_remote_code=True`
- Extract text and EUROVOC concept labels
- Build label vocabulary (map concept IDs to indices)
- Tokenize text with truncation to max_length
- Return: input_ids, attention_mask, labels (multi-hot)

**1.2 Data Collator**
- Pad sequences to batch max length
- Stack labels as multi-hot vectors

### Phase 2: Model Architecture

**2.1 Classification Model**
- Use transformer encoder (not decoder)
- Options:
  - Custom transformer encoder
  - Pretrained RoBERTa/BERT (smaller variant for budget)
- Add classification head: Linear(hidden_dim, num_labels)

**2.2 QuantizedClassifier Wrapper**
- Wrap encoder layers with FP8 quantization
- Preserve classification head in BF16 (critical for output)

### Phase 3: Training Loop

**3.1 Loss Function**
- Binary cross-entropy with logits
- Handle class imbalance (optional: focal loss)

**3.2 Metrics**
- F1 micro (global TP/FP/FN)
- F1 macro (average per-label F1)
- Precision, Recall
- Throughput (samples/sec)

**3.3 Trainer Refactor**
- Classification forward pass
- Multi-label evaluation
- Integrate rank monitoring per step
- Integrate flip tracking per step

### Phase 4: Experiment Infrastructure

**4.1 Config Schema**
```yaml
# Required fields
model:
  type: encoder  # encoder, not decoder
  hidden_dim: 768
  num_layers: 6
  num_heads: 12
  max_length: 512

data:
  dataset: eurlex
  max_examples: null  # null = all

training:
  batch_size: 16
  learning_rate: 2e-5
  max_steps: 5000
  eval_interval: 500

quantization:
  format: E5M2  # or BF16, E3M4, etc.
  quantize_encoder: true
  quantize_classifier: false  # keep classifier in BF16

optimizer:
  type: adamw  # or manifold_adamw
  manifold_mantissa_bits: 2  # for manifold_adamw
```

**4.2 Experiment Runner**
- Load config
- Initialize model, data, optimizer
- Train loop with eval
- Log all metrics to W&B
- Save checkpoints

**4.3 Experiment Matrix Configs**
Create configs for all 12 conditions:
- bf16_adamw.yaml
- bf16_manifold.yaml
- e5m2_adamw.yaml
- e5m2_manifold.yaml
- e3m4_adamw.yaml
- e3m4_manifold.yaml
- e1m6_adamw.yaml
- e1m6_manifold.yaml
- e0m7_adamw.yaml
- e0m7_manifold.yaml
- e7m0_adamw.yaml
- e7m0_manifold.yaml

### Phase 5: Analysis

**5.1 Results Aggregation**
- Pull all runs from W&B
- Compute mean/std across seeds (if multiple)
- Build comparison tables

**5.2 Key Analyses**
1. **Throughput comparison:** Which formats are fastest?
2. **Task performance:** Which formats achieve best F1?
3. **Manifold benefit:** For each format, does ManifoldAdamW improve F1?
4. **Rank stability:** Which formats show rank collapse?
5. **Failure modes:** Where does each format fail?

**5.3 Core Research Question**
- Compute: F1(manifold) - F1(standard) for each format
- Identify: Which format has largest positive delta?
- Correlate: Delta vs format properties (mantissa bits, dynamic range)

## File Changes Required

### New Files
- `altgrad/training/classifier.py` - Classification model
- `altgrad/training/classification_data.py` - EUR-Lex data loader
- `altgrad/training/classification_metrics.py` - F1 computation
- `altgrad/training/classification_trainer.py` - Training loop
- `experiments/configs/matrix/*.yaml` - 12 experiment configs

### Modified Files
- `altgrad/training/__init__.py` - Export new components
- `experiments/run_experiment.py` - Support classification mode

### Deprecated (keep for reference)
- `altgrad/training/model.py` - GPT LM model
- `altgrad/training/data.py` - LM data loading
- `altgrad/training/trainer.py` - LM training loop

## Execution Order

1. **Data pipeline** - Must work first
2. **Model** - Depends on data shapes
3. **Trainer** - Depends on model and data
4. **Configs** - Depends on trainer interface
5. **Analysis** - After experiments run

## Success Criteria

- [ ] EUR-Lex loads with labels correctly
- [ ] Classification model trains without NaN
- [ ] F1 metrics computed correctly
- [ ] BF16 baseline achieves reasonable F1 (sanity check)
- [ ] All 12 experiment configs run to completion
- [ ] W&B logs contain all required metrics
- [ ] Analysis answers: "Which FP8 format benefits most from ManifoldAdamW?"

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| Data pipeline | 1-2 hours |
| Model | 2-3 hours |
| Trainer | 2-3 hours |
| Configs | 1 hour |
| Testing | 1-2 hours |
| **Total** | **7-11 hours** |

---

*This plan treats the research with the seriousness it requires. Each component will be implemented correctly before moving to the next.*
