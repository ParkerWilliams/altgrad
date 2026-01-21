---
phase: 02-baseline-validation
plan: 01
subsystem: data
tags: [dataset, tokenization, tiktoken, memmap, batch-loading]

dependency-graph:
  requires: []
  provides: [eurlex-binary-data, get_batch-function]
  affects: [02-02, 02-03, 02-04, 02-05]

tech-stack:
  added: [tiktoken, datasets, numpy]
  patterns: [nanoGPT-style-memmap, uint16-token-storage]

key-files:
  created:
    - altgrad/training/__init__.py
    - altgrad/training/data.py
    - tests/test_data.py
    - data/eurlex/train.bin
    - data/eurlex/val.bin
  modified: []

decisions:
  - id: DS-001
    title: Use lex_glue/ecthr_a instead of multi_eurlex
    rationale: nlpaueb/multi_eurlex uses deprecated script-based format no longer supported by HuggingFace datasets library
    impact: Still European legal text (ECHR cases), similar characteristics for training experiments

metrics:
  duration: 8 min
  completed: 2026-01-21
---

# Phase 2 Plan 1: EurLex Data Preparation Summary

**One-liner:** European legal text (lex_glue/ecthr_a) tokenized with GPT-2 BPE to nanoGPT-style uint16 memmap files (~21M tokens total).

## What Was Built

### Data Preparation Module
Created `altgrad/training/data.py` with two core functions:

1. **`prepare_eurlex(data_dir, num_proc)`** - Tokenizes European legal text dataset
   - Uses lex_glue/ecthr_a (European Court of Human Rights cases)
   - GPT-2 BPE tokenization via tiktoken (50257 vocab)
   - Appends EOT token after each document
   - Writes uint16 memmap files for efficient loading

2. **`get_batch(split, data_dir, block_size, batch_size, device)`** - Random batch loading
   - Memory-mapped file access (no full load)
   - Returns (x, y) where y = x shifted by 1 token
   - Supports cpu/cuda/mps devices

### Binary Data Files
- `data/eurlex/train.bin`: 18,743,070 tokens (35.7 MB)
- `data/eurlex/val.bin`: 2,283,739 tokens (4.4 MB)

### Test Suite
Created `tests/test_data.py` with 14 tests covering:
- Binary file existence and uint16 dtype
- Token values within GPT-2 vocab range
- Batch tensor shapes (batch_size, block_size)
- Autoregressive target shift verification
- Device placement (cpu, cuda, mps)
- Reproducibility with seeded random state

## Decisions Made

### DS-001: Dataset Source Change
**Decision:** Use lex_glue/ecthr_a instead of nlpaueb/multi_eurlex

**Context:** The HuggingFace datasets library (v4.5.0) no longer supports script-based dataset loading. The nlpaueb/multi_eurlex dataset uses this deprecated format.

**Alternatives considered:**
1. Downgrade datasets library - Would break other dependencies
2. Use wikitext - Not legal text, different characteristics
3. Use lex_glue/ecthr_a - European legal text, compatible format

**Outcome:** lex_glue/ecthr_a provides European Court of Human Rights legal cases, maintaining the "jagged loss landscape" characteristic important for the project's gradient stability experiments.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing dependencies**
- **Found during:** Task 1 verification
- **Issue:** numpy, tiktoken, datasets not installed in venv
- **Fix:** Installed via pip: `pip install numpy tiktoken datasets`
- **Commit:** Part of task 1

**2. [Rule 1 - Bug] Dataset API incompatibility**
- **Found during:** Task 2 data preparation
- **Issue:** nlpaueb/multi_eurlex uses deprecated script-based format
- **Fix:** Switched to lex_glue/ecthr_a which uses Parquet format
- **Files modified:** altgrad/training/data.py
- **Commit:** beb1e84

## Verification Results

```bash
# Import verification
$ python -c "from altgrad.training.data import prepare_eurlex, get_batch; print('imports ok')"
imports ok

# Test results
$ pytest tests/test_data.py -v
13 passed, 1 skipped (CUDA not available)
```

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 6f0db8b | feat | Add EurLex data preparation module |
| beb1e84 | feat | Prepare European legal text data and add tests |

## Next Phase Readiness

### Ready for 02-02 (nanoGPT Model Integration)
- Binary data files in place at `data/eurlex/`
- `get_batch()` function ready for training loop integration
- uint16 format compatible with nanoGPT data loading pattern

### Dependencies Satisfied
- Train/val splits exist and are loadable
- Batch loading verified with correct shapes
- Token values within GPT-2 vocab range

### Potential Issues
- None identified
