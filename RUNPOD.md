# RunPod Setup Instructions

Quick setup for running altgrad experiments on a fresh RunPod instance.

## 1. Clone Repository

```bash
git clone https://github.com/ParkerWilliams/altgrad.git
cd altgrad
```

## 2. Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install altgrad and all dependencies
pip install -e .

# Verify installation
python -c "from altgrad.quantization import E5M2; print('OK')"
python -c "from altgrad.training import ClassificationTrainer; print('OK')"
```

## 3. Run Smoke Test (Quick Validation)

Test with minimal data to verify everything works:

```bash
python experiments/run_classification.py \
    --format BF16 \
    --optimizer adamw \
    --max-examples 100 \
    --max-steps 10 \
    --batch-size 8
```

Expected output: Should run 10 steps and print F1 metrics.

## 4. Run Full Experiments

### Single Condition
```bash
# BF16 baseline with AdamW
python experiments/run_classification.py --format BF16 --optimizer adamw

# E5M2 with ManifoldAdamW
python experiments/run_classification.py --format E5M2 --optimizer manifold
```

### Full Matrix (12 conditions)
```bash
python experiments/run_classification.py --full-matrix
```

### With W&B Logging
```bash
export WANDB_API_KEY=your_key_here
python experiments/run_classification.py --format E5M2 --optimizer manifold --wandb-project altgrad-classification
```

## 5. Common Issues

### Out of Memory
Reduce batch size:
```bash
python experiments/run_classification.py --format E5M2 --optimizer manifold --batch-size 8
```

### Slow Dataset Download
EUR-Lex (~1GB) downloads on first run. Subsequent runs use cache.
The DistilBERT model (~250MB) also downloads on first use.

### W&B Login Required
If you see W&B auth errors, either:
- Set `WANDB_API_KEY` environment variable
- Or run without W&B (default): omit `--wandb-project`

### CUDA Not Available
Check CUDA is properly configured:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 6. Expected Results

After running, you should see:
- Checkpoints in `checkpoints/{format}_{optimizer}/`
- Console output with F1 metrics per experiment
- W&B dashboard (if enabled)

## 7. Experiment Matrix

| Format | Optimizer | Description |
|--------|-----------|-------------|
| BF16 | adamw | Baseline (no FP8 quantization) |
| BF16 | manifold | Baseline with ManifoldAdamW |
| E5M2 | adamw | Standard FP8 |
| E5M2 | manifold | Standard FP8 + manifold |
| E3M4 | adamw | Balanced E/M |
| E3M4 | manifold | Balanced E/M + manifold |
| E1M6 | adamw | High precision |
| E1M6 | manifold | High precision + manifold |
| E0M7 | adamw | Fixed-point |
| E0M7 | manifold | Fixed-point + manifold |
| E7M0 | adamw | Powers of 2 only |
| E7M0 | manifold | Powers of 2 + manifold |

## 8. Key Metrics to Watch

1. **F1 micro**: Primary classification metric
2. **Stall ratio**: Fraction of updates that don't change FP8 representation
3. **Stable rank**: Weight matrix health indicator
4. **Collapse warnings**: Alerts for rank degradation
