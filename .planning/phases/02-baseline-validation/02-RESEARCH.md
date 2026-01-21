# Phase 02: Baseline Validation - Research

**Researched:** 2026-01-21
**Domain:** BF16/FP8 training validation with nanoGPT, EurLex, W&B monitoring
**Confidence:** HIGH

## Summary

This phase establishes the baseline training infrastructure for comparing BF16 and standard FP8 (E5M2) training. The research covers five key domains: nanoGPT integration, EurLex dataset handling, W&B logging and monitoring, checkpoint management, and FP32 shadow gradient comparison.

The standard approach is to fork nanoGPT's training loop rather than using it as a library, inject quantization hooks at forward/backward boundaries, use the `nlpaueb/multi_eurlex` dataset tokenized with tiktoken's GPT-2 BPE, log all metrics to W&B at every step with alerts for stability thresholds, and maintain FP32 model copy for gradient cosine similarity comparison.

**Primary recommendation:** Fork nanoGPT's train.py/model.py (~600 lines total), integrate altgrad.quantization at tensor operations, use W&B for all logging with programmatic alerts, implement save-on-anomaly checkpointing for debugging NaN events.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.9+ | Deep learning framework | Native support for BF16, autocast, GradScaler |
| nanoGPT | master | Transformer training loop | Minimal ~300 line train.py, full control, no abstraction overhead |
| wandb | Latest | Experiment tracking | Industry standard, rich logging, alerts, comparison dashboards |
| tiktoken | Latest | GPT-2 BPE tokenizer | 3-6x faster than alternatives, OpenAI standard |
| datasets (HuggingFace) | Latest | Dataset loading | Standard for multi_eurlex access |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | Latest | Progress bars | Training loop visualization |
| numpy | Latest | Data preparation | Token storage as uint16, binary file handling |
| pyyaml | Latest | Config files | YAML experiment configuration |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nanoGPT fork | PyTorch Lightning | Lightning adds abstraction overhead, harder to inject custom quantization |
| W&B | TensorBoard | TensorBoard lacks alerts, run resumption, comparison dashboards |
| tiktoken | HuggingFace tokenizers | tiktoken is 3-6x faster, cleaner API |
| multi_eurlex | EURLEX57K | multi_eurlex is better maintained, multi-language, HuggingFace hosted |

**Installation:**
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm pyyaml
```

## Architecture Patterns

### Recommended Project Structure
```
altgrad/
├── quantization/         # Phase 1 complete - formats, ops, scaling, diagnostics
├── training/             # NEW: Training infrastructure
│   ├── __init__.py
│   ├── config.py         # Experiment configuration dataclasses
│   ├── data.py           # EurLex data loading and tokenization
│   ├── model.py          # Fork of nanoGPT model.py with hooks
│   ├── trainer.py        # Main training loop with quantization injection
│   ├── metrics.py        # Gradient stats, stability metrics computation
│   ├── checkpoint.py     # Checkpoint save/load with quantization state
│   └── callbacks.py      # W&B logging, alerts, save-on-anomaly
└── experiments/          # Experiment configs and scripts
    └── configs/          # YAML experiment definitions

data/
└── eurlex/              # Prepared tokenized data
    ├── train.bin
    └── val.bin
```

### Pattern 1: nanoGPT Training Loop Integration

**What:** Fork nanoGPT's train.py with injection points for quantization
**When to use:** All training runs in this project

**Key injection points from nanoGPT train.py:**

```python
# Source: https://github.com/karpathy/nanoGPT/blob/master/train.py
# Line ~277-300: Forward/backward with gradient accumulation

for micro_step in range(gradient_accumulation_steps):
    # === INJECTION POINT 1: Pre-forward quantization ===
    if config.use_fp8:
        model = quantize_activations(model, fp8_format, scale_activations)

    with ctx:  # autocast context for BF16
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps

    # === INJECTION POINT 2: Post-forward, pre-backward ===
    # Log forward stability metrics (overflow/underflow)

    scaler.scale(loss).backward()

    # === INJECTION POINT 3: Post-backward gradient processing ===
    # Compute gradient statistics before optimizer step
    # Compare FP8 gradients to FP32 shadow gradients

# Gradient clipping
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# === INJECTION POINT 4: Pre-optimizer step ===
# Detect bit-stall before weights update

scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)

# === INJECTION POINT 5: Post-optimizer step ===
# Log weight update statistics
```

### Pattern 2: EurLex Data Preparation (nanoGPT-style)

**What:** Convert EurLex to tokenized binary format for efficient loading
**When to use:** Dataset preparation before training

```python
# Source: Derived from nanoGPT/data/openwebtext/prepare.py pattern
import tiktoken
import numpy as np
from datasets import load_dataset

def prepare_eurlex():
    """Prepare EurLex dataset in nanoGPT binary format."""
    # Load dataset from HuggingFace
    dataset = load_dataset('nlpaueb/multi_eurlex', 'en')

    # Initialize GPT-2 BPE tokenizer
    enc = tiktoken.get_encoding('gpt2')  # 50257 vocab

    def tokenize(example):
        # Encode text, add EOT token
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)  # End of text
        return {'ids': ids, 'len': len(ids)}

    # Tokenize all documents
    tokenized = dataset.map(tokenize, num_proc=8)

    # Concatenate all tokens into single array
    for split, dset in tokenized.items():
        arr_len = sum(dset['len'])
        filename = f'data/eurlex/{split}.bin'
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))

        idx = 0
        for example in dset:
            ids = np.array(example['ids'], dtype=np.uint16)
            arr[idx:idx+len(ids)] = ids
            idx += len(ids)
        arr.flush()
```

**Key points:**
- Tokens stored as uint16 (GPT-2 vocab is 50257, fits in 16 bits)
- Memory-mapped files for efficient loading
- EOT token separates documents

### Pattern 3: W&B Every-Step Logging with Alerts

**What:** Log all metrics every step, use programmatic alerts for threshold breaches
**When to use:** All training runs

```python
# Source: https://docs.wandb.ai/guides/track/log/ + https://docs.wandb.ai/models/runs/alert
import wandb
from wandb import AlertLevel

class WandbTracker:
    """Comprehensive W&B logging with alerts."""

    def __init__(self, config, run_id=None):
        self.run = wandb.init(
            project="altgrad",
            config=vars(config),
            id=run_id,
            resume="allow" if run_id else None,
        )
        self.consecutive_nan_steps = 0
        self.loss_history = []

    def log_step(self, step: int, metrics: dict):
        """Log all metrics for a training step."""
        # Log with explicit step number for precise control
        wandb.log(metrics, step=step)

    def check_alerts(self, step: int, metrics: dict, config):
        """Check threshold breaches and send alerts."""

        # NaN detection
        if metrics.get('loss_is_nan', False):
            self.consecutive_nan_steps += 1
            if self.consecutive_nan_steps >= config.nan_patience:
                self.run.alert(
                    title="NaN Detected - Auto-stopping",
                    text=f"Step {step}: {config.nan_patience} consecutive NaN losses",
                    level=AlertLevel.ERROR,
                )
                return 'stop'
            self.run.alert(
                title="NaN Warning",
                text=f"Step {step}: NaN in loss ({self.consecutive_nan_steps} consecutive)",
                level=AlertLevel.WARN,
                wait_duration=60,  # Max 1 alert per minute
            )
            return 'save_checkpoint'  # Save for debugging
        else:
            self.consecutive_nan_steps = 0

        # Bit-stall warning
        if metrics.get('bit_stall_rate', 0) > config.bit_stall_threshold:
            self.run.alert(
                title="High Bit-Stall Rate",
                text=f"Step {step}: {metrics['bit_stall_rate']:.1%} > {config.bit_stall_threshold:.1%}",
                level=AlertLevel.WARN,
                wait_duration=300,  # Max 1 alert per 5 minutes
            )

        # Overflow/underflow warning
        if metrics.get('overflow_rate', 0) > config.overflow_threshold:
            self.run.alert(
                title="High Overflow Rate",
                text=f"Step {step}: {metrics['overflow_rate']:.1%} overflow",
                level=AlertLevel.WARN,
                wait_duration=300,
            )

        return 'continue'
```

### Pattern 4: FP32 Shadow for Gradient Comparison

**What:** Maintain FP32 copy for per-step gradient cosine similarity
**When to use:** FP8 training runs (not BF16 baseline)

```python
# Source: Mixed precision training papers + PyTorch CosineSimilarity
import torch
import torch.nn.functional as F

class FP32ShadowModel:
    """Maintains FP32 copy for gradient comparison."""

    def __init__(self, model: torch.nn.Module):
        # Deep copy model to FP32
        self.shadow_model = copy.deepcopy(model)
        self.shadow_model.to(torch.float32)
        # Disable gradient tracking on shadow parameters
        for p in self.shadow_model.parameters():
            p.requires_grad = True

    def sync_weights(self, quantized_model):
        """Copy quantized model weights to shadow (dequantized)."""
        with torch.no_grad():
            for (name, p_quant), (_, p_shadow) in zip(
                quantized_model.named_parameters(),
                self.shadow_model.named_parameters()
            ):
                # Copy dequantized values
                p_shadow.data.copy_(p_quant.data.float())

    def compute_gradient_similarity(self, quantized_model) -> dict:
        """Compare FP8 gradients to FP32 reference gradients."""
        similarities = {}

        for (name, p_quant), (_, p_shadow) in zip(
            quantized_model.named_parameters(),
            self.shadow_model.named_parameters()
        ):
            if p_quant.grad is None or p_shadow.grad is None:
                continue

            # Flatten gradients for cosine similarity
            g_quant = p_quant.grad.detach().flatten().float()
            g_shadow = p_shadow.grad.detach().flatten()

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(g_quant.unsqueeze(0), g_shadow.unsqueeze(0))
            similarities[f'grad_cos_sim/{name}'] = cos_sim.item()

        # Aggregate: mean and min across layers
        values = list(similarities.values())
        similarities['grad_cos_sim/mean'] = sum(values) / len(values) if values else 0
        similarities['grad_cos_sim/min'] = min(values) if values else 0

        return similarities
```

**Memory consideration:** Shadow model doubles parameter memory but NOT activation memory (gradients computed separately, not stored simultaneously).

### Pattern 5: Checkpoint with Quantization State

**What:** Save model + optimizer + quantization state for resumable training
**When to use:** Every N steps + on-anomaly save

```python
# Source: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
import torch

def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: dict,
    quantization_state: dict = None,
):
    """Save complete training state including quantization."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'config': config,
        # Quantization-specific state
        'quantization_state': quantization_state or {},
        # For reproducibility
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath: str, model, optimizer, scaler=None):
    """Load checkpoint and restore full training state."""
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Restore RNG state for reproducibility
    rng = checkpoint.get('rng_state', {})
    if rng.get('python'):
        random.setstate(rng['python'])
    if rng.get('numpy'):
        np.random.set_state(rng['numpy'])
    if rng.get('torch'):
        torch.set_rng_state(rng['torch'])
    if rng.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])

    return checkpoint['step'], checkpoint['config'], checkpoint.get('quantization_state', {})
```

**Quantization state to save:**
- `amax_history` buffers per quantized tensor
- `scale_factors` current values
- `bit_stall_counters` for debugging

### Pattern 6: Per-Layer Gradient Statistics

**What:** Compute gradient norms, dead neuron fraction, SNR per layer
**When to use:** Every step for full visibility

```python
# Source: https://neptune.ai/blog/monitoring-diagnosing-and-solving-gradient-issues-in-foundation-models
import torch

def compute_gradient_stats(model: torch.nn.Module, threshold: float = 1e-8) -> dict:
    """Compute per-layer gradient statistics."""
    stats = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()

        # L2 and Linf norms
        stats[f'grad_norm_l2/{name}'] = grad.norm(2).item()
        stats[f'grad_norm_linf/{name}'] = grad.abs().max().item()

        # Dead neuron fraction (gradient below threshold)
        dead_fraction = (grad.abs() < threshold).float().mean().item()
        stats[f'dead_neuron_frac/{name}'] = dead_fraction

        # Signal-to-noise ratio (mean / std)
        mean = grad.mean().item()
        std = grad.std().item()
        snr = abs(mean) / (std + 1e-10)
        stats[f'grad_snr/{name}'] = snr

    return stats
```

### Anti-Patterns to Avoid

- **Logging inside micro-step:** Only log after full gradient accumulation, not each micro-step
- **Gradient comparison with shared graph:** FP32 shadow must use separate backward(), not share computation
- **Blocking W&B uploads:** Use background uploading, don't call `wandb.finish()` until training complete
- **Saving all checkpoints:** Only retain best + last N, delete intermediate to save disk
- **Per-step checkpoint saves:** Only save every N steps (100) plus on-anomaly

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GPT-2 tokenization | Custom BPE | tiktoken | 3-6x faster, handles edge cases, 50257 vocab |
| Cosine LR schedule | Manual decay | torch.optim.lr_scheduler.CosineAnnealingLR | Handles edge cases, checkpointable state |
| Mixed precision scaling | Manual loss scaling | torch.amp.GradScaler | Automatic scale adjustment, overflow detection |
| Run comparison | Manual metrics aggregation | W&B comparison view | Built-in dashboards, filtering, tagging |
| Gradient clipping | Manual norm computation | torch.nn.utils.clip_grad_norm_ | Handles multiple parameter groups correctly |
| Checkpoint versioning | Manual file naming | W&B Artifacts or simple best+last-N scheme | Version tracking, aliasing (best, latest) |

**Key insight:** nanoGPT's ~300 line train.py already handles learning rate warmup, gradient accumulation, mixed precision, and checkpointing. Fork and extend, don't rewrite.

## Common Pitfalls

### Pitfall 1: W&B Step Mismatch Across Metrics

**What goes wrong:** Metrics logged at different points have inconsistent step numbers
**Why it happens:** Multiple `wandb.log()` calls without explicit step, auto-increment
**How to avoid:** Always pass explicit `step=` parameter, batch all metrics per step into single dict

**Warning signs:**
- Charts show jagged lines or gaps
- X-axis doesn't align between metrics
- "Step" in UI shows different values than training log

**Prevention strategy:**
```python
# CORRECT: Single log call with explicit step
metrics = {**loss_metrics, **grad_metrics, **stability_metrics}
wandb.log(metrics, step=global_step)

# WRONG: Multiple log calls with auto-increment
wandb.log(loss_metrics)
wandb.log(grad_metrics)  # Step is now global_step + 1
```

### Pitfall 2: Shadow Model Gradient Accumulation

**What goes wrong:** Shadow gradients accumulate across steps, comparison is wrong
**Why it happens:** Forgetting to zero_grad() on shadow model
**How to avoid:** Always zero shadow gradients before shadow backward pass

**Warning signs:**
- Gradient cosine similarity decreases over time
- Shadow gradient norms grow unboundedly
- Comparison shows FP8 gradients as "better" than FP32

**Prevention strategy:**
```python
# Run FP32 shadow forward/backward with same input
shadow_optimizer.zero_grad()
shadow_logits, shadow_loss = shadow_model(X, Y)
shadow_loss.backward()
# Now compare gradients
similarity = compute_gradient_similarity(model, shadow_model)
```

### Pitfall 3: EurLex Document Length Variance

**What goes wrong:** Very long documents cause OOM, short documents waste compute
**Why it happens:** Legal documents range from hundreds to tens of thousands of words
**How to avoid:** Use fixed block_size (e.g., 1024 tokens), documents are concatenated and chunked

**Warning signs:**
- CUDA OOM on some batches but not others
- Highly variable batch processing time
- Loss spikes on certain batches

**Prevention strategy:**
```python
# nanoGPT style: all tokens concatenated, sampled in fixed blocks
# No per-document batching, just continuous token stream
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x, y
```

### Pitfall 4: W&B Resume Creates New Run

**What goes wrong:** Run resume creates a new run instead of continuing existing
**Why it happens:** Wrong `resume=` mode or missing run ID
**How to avoid:** Use `resume="must"` with explicit `id=` when continuing

**Warning signs:**
- Dashboard shows multiple runs instead of one continuous run
- Step numbers reset to 0 after restart
- Metrics history appears truncated

**Prevention strategy:**
```python
# Save run ID on first init
run = wandb.init(project="altgrad", ...)
run_id = run.id
# Save run_id to checkpoint

# On resume
run = wandb.init(
    project="altgrad",
    id=saved_run_id,
    resume="must",  # Fail if run doesn't exist
)
```

### Pitfall 5: Identical Seeds Not Producing Identical Results

**What goes wrong:** BF16 and FP8 runs with same seed diverge earlier than expected
**Why it happens:** Quantization adds non-determinism, or seed not applied to all RNG sources
**How to avoid:** Set ALL RNG sources, use `torch.use_deterministic_algorithms(True)` if strict comparison needed

**Warning signs:**
- Runs diverge on step 1 instead of gradually
- Same seed gives different loss on repeat runs
- Impossible to reproduce specific failure

**Prevention strategy:**
```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For strictest reproducibility (may slow training):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
```

### Pitfall 6: NaN Check Before vs After Backward

**What goes wrong:** NaN detection misses forward-pass NaNs or triggers false positives
**Why it happens:** Loss can be valid but gradients are NaN, or vice versa
**How to avoid:** Check both loss and gradients, log which is the source

**Warning signs:**
- NaN alert but loss looks fine in logs
- Training crashes but no NaN warning was triggered
- Checkpoint saved with corrupted weights

**Prevention strategy:**
```python
# Check loss after forward
loss_is_nan = torch.isnan(loss).any().item()

# Check gradients after backward
grad_has_nan = any(
    torch.isnan(p.grad).any().item()
    for p in model.parameters()
    if p.grad is not None
)

# Log both
wandb.log({
    'loss_is_nan': loss_is_nan,
    'grad_has_nan': grad_has_nan,
}, step=step)
```

## Code Examples

Verified patterns from official sources:

### nanoGPT-Style Data Loading
```python
# Source: https://github.com/karpathy/nanoGPT/blob/master/train.py
import numpy as np
import torch

def get_batch(split: str, data_dir: str, block_size: int, batch_size: int, device: str):
    """Load batch from memory-mapped binary file."""
    data = np.memmap(f'{data_dir}/{split}.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i+block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
        for i in ix
    ])
    x, y = x.to(device), y.to(device)
    return x, y
```

### W&B Alert for Stability Threshold
```python
# Source: https://docs.wandb.ai/models/runs/alert
import wandb
from wandb import AlertLevel

def check_stability_alert(metrics: dict, step: int, run: wandb.sdk.wandb_run.Run):
    """Send W&B alert on stability threshold breach."""

    # Bit-stall threshold: 50%
    if metrics.get('bit_stall_rate', 0) > 0.5:
        run.alert(
            title="High Bit-Stall Rate",
            text=f"Step {step}: {metrics['bit_stall_rate']:.1%} stall rate exceeds 50% threshold",
            level=AlertLevel.WARN,
            wait_duration=300,  # Cooldown between alerts
        )

    # Overflow threshold: 1%
    if metrics.get('overflow_rate', 0) > 0.01:
        run.alert(
            title="High Overflow Rate",
            text=f"Step {step}: {metrics['overflow_rate']:.2%} overflow rate exceeds 1% threshold",
            level=AlertLevel.WARN,
            wait_duration=300,
        )
```

### Gradient Cosine Similarity Computation
```python
# Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
import torch
import torch.nn.functional as F

def gradient_cosine_similarity(model_a, model_b) -> float:
    """Compute cosine similarity between gradients of two models."""
    grads_a = []
    grads_b = []

    for (_, p_a), (_, p_b) in zip(
        model_a.named_parameters(),
        model_b.named_parameters()
    ):
        if p_a.grad is not None and p_b.grad is not None:
            grads_a.append(p_a.grad.flatten())
            grads_b.append(p_b.grad.flatten())

    if not grads_a:
        return 0.0

    # Concatenate all gradients into single vectors
    flat_a = torch.cat(grads_a).float()
    flat_b = torch.cat(grads_b).float()

    # Compute cosine similarity
    similarity = F.cosine_similarity(
        flat_a.unsqueeze(0),
        flat_b.unsqueeze(0)
    ).item()

    return similarity
```

### Perplexity Calculation for Language Modeling
```python
# Source: Standard definition: PPL = exp(cross_entropy_loss)
import torch
import math

def compute_perplexity(loss: torch.Tensor) -> float:
    """Compute perplexity from cross-entropy loss."""
    # loss is already cross-entropy (negative log likelihood)
    # perplexity = exp(loss)
    return math.exp(loss.item())

# In training loop:
with torch.no_grad():
    logits, loss = model(val_x, val_y)
    val_ppl = compute_perplexity(loss)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-run checkpointing | W&B Artifacts with aliasing | 2023-2024 | Automatic versioning, "best"/"latest" aliases |
| Cosine LR only | Trapezoidal LR schedule | 2025 (nanoGPT speedrun) | Easier to tune, matches cosine performance |
| Manual loss scaling | torch.amp.GradScaler auto-scaling | 2023+ | Automatic scale adjustment on overflow |
| Log every N steps | Log every step | Research mode | Full visibility critical for debugging quantization |
| Per-layer amax | Per-tensor and per-block scaling | 2024-2025 (MXFP8) | Per-tensor standard for training, per-block for extreme cases |

**Deprecated/outdated:**
- Manual FP16 scaling: Use GradScaler
- TensorBoard for ML experiments: W&B provides better comparison and collaboration
- Fixed LR schedules: Warmup + decay is standard

**Bleeding edge (2026):**
- MXFP8 (microscaling): 32-element blocks with E8M0 scale factors
- torch.compile() for quantization: Better performance through fusion
- W&B Multi-video sync: Qualitative comparison for generated content

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal EurLex subset size for convergence trends**
   - What we know: Full dataset is 65k documents, ~10M+ tokens
   - What's unclear: How many steps/tokens needed to see convergence divergence between BF16/FP8
   - Recommendation: Start with 1-5% of dataset, ~100k tokens, tune based on validation loss curves

2. **FP32 shadow memory overhead on H100**
   - What we know: Shadow doubles parameter memory, not activations
   - What's unclear: For 50M parameter limit, is 100M parameters (model + shadow) feasible
   - Recommendation: 50M params * 4 bytes * 2 = 400MB, well within H100 80GB, proceed with full shadow

3. **W&B alert rate limiting for high-frequency events**
   - What we know: `wait_duration` parameter limits alert frequency
   - What's unclear: Optimal cooldown to balance awareness vs. alert fatigue
   - Recommendation: 60s for critical (NaN), 300s for warnings (bit-stall, overflow)

4. **Gradient comparison frequency vs. overhead**
   - What we know: Per-step comparison is desired (from CONTEXT.md)
   - What's unclear: Overhead of shadow forward/backward every step
   - Recommendation: Implement per-step, measure overhead, add config option to reduce if needed

5. **Dead neuron persistence window**
   - What we know: Need to track gradients below 1e-8 "for N steps"
   - What's unclear: What value of N indicates a truly dead neuron vs. temporary inactivity
   - Recommendation: Start with N=100 steps, make configurable, log distribution

## Sources

### Primary (HIGH confidence)

**nanoGPT:**
- [GitHub Repository](https://github.com/karpathy/nanoGPT) - train.py structure, configuration system
- [train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py) - ~300 line training loop reference

**EurLex Dataset:**
- [nlpaueb/multi_eurlex on HuggingFace](https://huggingface.co/datasets/nlpaueb/multi_eurlex) - Dataset card, loading instructions
- [MultiEURLEX Paper (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.559/) - Dataset description and benchmarks

**W&B:**
- [Logging Guide](https://docs.wandb.ai/guides/track/log/) - Step management, metric naming
- [Alert API](https://docs.wandb.ai/models/runs/alert) - run.alert() parameters and usage
- [Resume Guide](https://docs.wandb.ai/models/runs/resuming) - Run resumption patterns

**PyTorch:**
- [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) - Checkpoint best practices
- [torch.autograd.set_detect_anomaly](https://docs.pytorch.org/docs/stable/autograd.html) - NaN detection
- [torch.nn.CosineSimilarity](https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html) - Gradient comparison

### Secondary (MEDIUM confidence)

- [tiktoken GitHub](https://github.com/openai/tiktoken) - GPT-2 tokenizer performance claims
- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) - FP32 master weights rationale
- [Neptune.ai Gradient Monitoring](https://neptune.ai/blog/monitoring-diagnosing-and-solving-gradient-issues-in-foundation-models) - Per-layer gradient tracking patterns
- [PyTorch Dead ReLU Discussion](https://discuss.pytorch.org/t/how-to-monitor-the-dead-neuron-during-training-process/3837) - Dead neuron detection approaches

### Tertiary (LOW confidence - marked for validation)

- [NaN Capture Callback (Medium)](https://chaimrand.medium.com/debugging-the-dreaded-nan-ac3f9feac5b2) - Save-on-anomaly pattern (verify implementation)
- nanoGPT speedrun leaderboard improvements - Trapezoidal LR claims (verify with official repo)

## Metadata

**Confidence breakdown:**
- nanoGPT integration: HIGH - Official repo, well-documented ~300 line files
- EurLex dataset: HIGH - HuggingFace hosted, standard load_dataset API
- W&B logging: HIGH - Official docs, well-established API
- Checkpoint management: HIGH - PyTorch official tutorials
- FP32 shadow approach: MEDIUM - Standard in mixed precision, implementation details vary
- Dead neuron thresholds: LOW - No established standard, requires empirical tuning

**Research date:** 2026-01-21

**Valid until:** 30 days for stable APIs (PyTorch, W&B), 7 days for nanoGPT (active development)

**Implementation status:**
- altgrad.quantization: COMPLETE (Phase 1)
- nanoGPT fork: TODO
- EurLex preparation: TODO
- W&B integration: TODO
- Checkpoint system: TODO
- Gradient comparison: TODO
- Training metrics: TODO

**Key takeaway for planner:** The infrastructure is well-established. Focus on:
1. Forking nanoGPT (~600 lines model.py + train.py)
2. Preparing EurLex in nanoGPT binary format
3. Injecting altgrad.quantization at forward/backward boundaries
4. Wiring up comprehensive W&B logging with alerts
5. Implementing FP32 shadow for gradient comparison
6. Running BF16 baseline, then E5M2 FP8 with identical seeds
