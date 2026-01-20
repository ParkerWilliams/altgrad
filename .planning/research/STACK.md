# Stack Research: FP8 Quantized Training Test Bench

**Project:** AltGrad - FP8 Training Test Bench
**Domain:** Custom floating-point format experimentation & manifold-aware optimization
**Researched:** January 20, 2026
**Target Hardware:** Single H100 GPU

---

## Executive Summary

Building an FP8 training test bench in 2025-2026 requires choosing between PyTorch's maturing native FP8 support (via TorchAO) and custom implementations. For your use case—experimenting with non-standard FP8 formats (E1M6, E3M4, E7M0) beyond PyTorch's native E4M3/E5M2—you'll need **custom FP8 operations** while leveraging PyTorch's native types where possible.

**Recommended approach:** Start with PyTorch 2.8's native FP8 types for standard formats, implement custom operations using `torch.library.custom_op` for exotic formats, and use geoopt for manifold-aware optimizer experiments.

---

## Recommended Core Stack

### Deep Learning Framework

| Technology | Version | Purpose | Rationale |
|------------|---------|---------|-----------|
| **PyTorch** | 2.8+ | Core framework | Latest stable with mature FP8 support (torch.float8_e4m3fn, torch.float8_e5m2), custom_op API for exotic formats, and excellent H100 support. Released Aug 2025. |
| **Python** | 3.12 | Runtime | Recommended by NVIDIA Transformer Engine, good ecosystem support |
| **CUDA** | 12.6+ | GPU acceleration | 12.8+ has better FP8 Tensor Core support but dropped sm50-60. Use 12.6 for broader compat, 12.8+ for latest features. |

**Why PyTorch 2.8:**
- Native FP8 types (E4M3fn, E5M2) since 2.2, matured through 2.8
- `torch.library.custom_op` API for implementing custom FP8 formats
- `torch._scaled_mm()` for scaled matrix multiplications with FP8
- `torch.compile()` integration for JIT optimization
- Excellent H100 FP8 Tensor Core utilization

**Confidence:** HIGH (verified via official PyTorch 2.8 release notes and NVIDIA documentation)

---

## FP8 Quantization Stack

### Option 1: TorchAO (Recommended for Standard FP8)

| Component | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| **TorchAO** | 0.10+ (nightly recommended) | FP8 training with E4M3/E5M2 | Baseline experiments with standard formats, performance comparisons |

**Installation:**
```bash
pip install torchao  # stable
# OR for latest features:
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu124
```

**Key Features:**
- `convert_to_float8_training(model)` - one-line FP8 conversion
- Tensorwise, rowwise, and MXFP8 (block-scaled) quantization modes
- Native integration with `torch.compile()`, FSDP2, tensor parallelism
- **Performance:** 1.5x speedup on LLaMA-3.1-70B, 1.28x on Llama3-70B with MXFP8
- Proven at scale (tested on 1856 GPUs / 2k GPU clusters)

**Usage Pattern:**
```python
from torchao.float8 import convert_to_float8_training

# Apply to model (uses E4M3 for weights/activations, E5M2 for gradients)
model = convert_to_float8_training(model)
```

**Limitations:**
- **Only supports E4M3fn and E5M2** (PyTorch native types)
- Cannot experiment with E1M6, E3M4, E5M2, E7M0 variations
- Scaling strategies are predefined (tensorwise, rowwise, block-scaled)

**Confidence:** HIGH (verified via TorchAO GitHub, PyTorch blog, production deployment reports)

---

### Option 2: Custom FP8 Operations (Required for Exotic Formats)

For non-standard formats (E1M6, E3M4, E7M0), you'll need custom implementations.

| Component | Version | Purpose | Rationale |
|-----------|---------|---------|-----------|
| **PyTorch custom_op API** | PyTorch 2.8+ | Define custom FP8 matmul operations | Proven in modded-nanoGPT for custom FP8 ops on H100 |
| **torch._scaled_mm** | PyTorch 2.8+ | Hardware-accelerated scaled matrix multiply | Leverages H100 FP8 Tensor Cores directly |

**Implementation Approach (based on modded-nanoGPT):**

```python
import torch
from torch.library import custom_op

@custom_op("mylib::fp8_mm", mutates_args=())
def fp8_mm(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Custom FP8 matrix multiply with arbitrary E*M* format."""
    # 1. Quantize to custom FP8 format (you implement this)
    x_fp8 = quantize_to_custom_fp8(x, x_scale, exp_bits=4, man_bits=3)
    w_fp8 = quantize_to_custom_fp8(w, w_scale, exp_bits=4, man_bits=3)

    # 2. Use PyTorch's scaled_mm if format matches E4M3/E5M2
    # OR implement custom CUDA kernel for exotic formats
    result = torch._scaled_mm(x_fp8, w_fp8, scale_a=x_scale, scale_b=w_scale)

    return result

@fp8_mm.register_fake
def _(x, w, x_scale, w_scale):
    """Fake implementation for torch.compile tracing."""
    return torch.empty(x.shape[0], w.shape[1], dtype=torch.bfloat16, device=x.device)

@fp8_mm.register_autograd
def _(ctx, saved, grad_out):
    """Backward pass with E5M2 for gradients."""
    # Implement gradient computation
    pass
```

**For truly exotic formats (E1M6, E3M4, E7M0):**

Since PyTorch only natively supports E4M3fn and E5M2, you'll need to:
1. **Emulate via INT8 storage** + custom quantize/dequantize functions
2. **Implement CUDA kernels** for actual hardware acceleration (significant effort)
3. **Use straight-through estimators (STE)** for gradients

**Confidence:** MEDIUM (modded-nanoGPT demonstrates viability for E4M3/E5M2; exotic formats will require substantial custom work)

---

### Option 3: QPyTorch (For Simulation Only)

| Component | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| **QPyTorch** | 0.0.1 alpha (archived) | Low-precision simulation | Prototyping exotic formats WITHOUT hardware acceleration |

**Installation:**
```bash
pip install git+https://github.com/Tiiiger/QPyTorch.git
```

**Features:**
- Simulate arbitrary floating-point formats: `FloatingPoint(exp=5, man=2)`
- Define different formats for forward/backward passes
- Quantizer wrappers for layers and optimizers

**Example:**
```python
from qtorch.quant import Quantizer, quantizer
from qtorch import FloatingPoint

# Define custom format
bit_8_e1m6 = FloatingPoint(exp=1, man=6)
weight_quant = quantizer(forward_number=bit_8_e1m6, forward_rounding="nearest")

# Wrap layers
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        layer.weight = weight_quant(layer.weight)
```

**Critical Limitations:**
- **Software simulation only** - no hardware acceleration
- Emulates FP8 using FP32 arithmetic (massive slowdown)
- Last updated 2019, PyTorch 1.5.0 compatibility
- **Not suitable for your use case** (you need H100 acceleration)

**Recommendation:** Use only for initial format exploration, then migrate to custom ops.

**Confidence:** MEDIUM (library exists but is unmaintained; functionality verified in papers but not actively developed)

---

### Option 4: NVIDIA Transformer Engine (Alternative)

| Component | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| **Transformer Engine** | Latest (v2.11+) | FP8 training for transformers | If abandoning custom formats, maximizing E4M3/E5M2 performance |

**Installation:**
```bash
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

**Features:**
- Drop-in FP8 transformer layers (`te.Linear`, `te.LayerNorm`)
- Automatic scaling and format selection (E4M3 forward, E5M2 backward)
- FP8 attention kernels optimized for H100
- **20-30% speedup** over PyTorch FP8 in some configs

**Limitations:**
- **Only supports E4M3/E5M2** (same as TorchAO)
- Transformer-specific (less flexible for custom architectures)
- Requires Hopper+ GPUs (H100/B200, not A100)

**When to use:** If your test bench evolves into a production FP8 transformer training system.

**When NOT to use:** For custom FP8 format experimentation.

**Confidence:** HIGH (NVIDIA official library, extensively documented)

---

## Optimizer & Manifold Optimization Stack

### Natural Gradient / Fisher Information Libraries

| Library | Version | Purpose | Rationale |
|---------|---------|---------|-----------|
| **NNGeometry** | 0.4 (Apr 2025) | Fisher Information Matrix computation | Active, modern, supports KFAC/EKFAC/diagonal approximations |
| **Geoopt** | 0.5.1 (Jun 2023) | Riemannian optimization | Manifold-aware optimizers (RiemannianSGD, RiemannianAdam) |

---

### NNGeometry - Fisher Information Matrix

**Installation:**
```bash
pip install nngeometry
# OR from source:
pip install git+https://github.com/tfjgeorge/nngeometry.git
```

**Key Capabilities:**
- **FIM representations:** Full, diagonal, KFAC, EKFAC, low-rank, implicit
- **Efficient computation:** No explicit large matrix storage
- **NTK support:** Neural Tangent Kernel evaluation
- **Jacobian computation:** Per-sample gradients w.r.t. parameters

**Usage Pattern:**
```python
from nngeometry.object import PMatKFAC
from nngeometry.metrics import FIM

# Compute KFAC Fisher Information Matrix
F = FIM(model=model,
        loader=train_loader,
        representation=PMatKFAC,
        n_output=10)

# Use for natural gradient descent
natural_grad = F.solve(grad)  # F^-1 @ grad
```

**For Your Use Case:**
- Compute Fisher w.r.t. FP8-quantized gradients
- Implement manifold-aware preconditioner for FP8 space
- Analyze curvature in quantized parameter space

**Confidence:** HIGH (active development, verified via GitHub releases and documentation)

---

### Geoopt - Riemannian Optimization

**Installation:**
```bash
pip install geoopt
# OR latest:
pip install git+https://github.com/geoopt/geoopt.git
```

**Key Capabilities:**
- **ManifoldParameter:** Drop-in replacement for torch.Parameter with manifold constraints
- **Riemannian optimizers:** RiemannianSGD, RiemannianAdam
- **Manifold operations:** `.egrad2rgrad(u)` projects Euclidean gradient to Riemannian gradient
- **PyTorch integration:** Works with `torch.nn.Module`, autograd, `torch.compile()`

**Usage Pattern:**
```python
import geoopt

# Define manifold (or create custom manifold for FP8 space)
manifold = geoopt.manifolds.Euclidean()  # Replace with custom FP8 manifold

# Create manifold parameter
param = geoopt.ManifoldParameter(
    torch.randn(10, 10),
    manifold=manifold
)

# Use Riemannian optimizer
optimizer = geoopt.optim.RiemannianAdam([param], lr=1e-3)
```

**For Your Use Case - FP8 Manifold:**

You'll need to **implement a custom Manifold class** for FP8 quantization space:

```python
from geoopt import Manifold

class FP8Manifold(Manifold):
    """
    Manifold representing FP8 quantization space.
    Treats uniform bit-space as warped manifold of representable values.
    """

    def __init__(self, exp_bits, man_bits):
        super().__init__()
        self.exp_bits = exp_bits
        self.man_bits = man_bits
        # Precompute representable FP8 values and their density
        self.values = self._compute_fp8_grid()
        self.metric = self._compute_stiffness_metric()

    def _compute_fp8_grid(self):
        """Enumerate all 2^8 representable FP8 values."""
        # Generate all possible bit patterns
        pass

    def _compute_stiffness_metric(self):
        """
        Compute metric tensor encoding density of representable values.
        High density regions (near zero) have high stiffness.
        """
        # Stiffness ∝ d(bitspace)/d(value)
        pass

    def projx(self, x):
        """Project continuous value to nearest FP8 value."""
        # Round to nearest representable value
        pass

    def egrad2rgrad(self, x, u):
        """
        Convert Euclidean gradient to Riemannian gradient.
        Applies stiffness-based preconditioner.
        """
        # Scale gradient by inverse metric tensor
        preconditioned_grad = self.metric.inverse() @ u
        return preconditioned_grad

    def retr(self, x, u):
        """Retraction: move along geodesic, then project."""
        # x_new = projx(x + u)
        pass
```

**PyTorch Compatibility:** Requires PyTorch 2.0+, officially supports 2 latest stable versions.

**Confidence:** HIGH (active library, well-documented API)

---

### Alternative: wiseodd/natural-gradients (Reference Implementation)

| Component | Version | Purpose | When to Use |
|-----------|---------|---------|-------------|
| **wiseodd/natural-gradients** | GitHub repo | Reference implementations of EKFAC, K-FAC, etc. | Learning/prototyping natural gradient algorithms |

**Installation:**
```bash
# Clone and adapt code
git clone https://github.com/wiseodd/natural-gradients.git
```

**Features:**
- Reference PyTorch implementations of EKFAC, K-FAC, Natural Adam
- Educational code (not production library)

**Use case:** Study natural gradient algorithms, adapt for FP8 context.

**Confidence:** MEDIUM (reference implementations, not production-ready library)

---

## Dataset Stack

### EurLex Legal Document Classification

| Component | Version | Purpose | Rationale |
|-----------|---------|---------|-----------|
| **Hugging Face Datasets** | Latest | Dataset loading | Standard interface for nlpaueb/multi_eurlex |
| **nlpaueb/multi_eurlex** | v1.0 | Legal document classification | 65k EU laws, multi-label, multiple EUROVOC levels |

**Installation:**
```bash
pip install datasets transformers
```

**Loading:**
```python
from datasets import load_dataset

# Load MultiEURLEX dataset
dataset = load_dataset("nlpaueb/multi_eurlex", "en")  # English split

# Structure:
# - 'train': training documents
# - 'test': test documents
# - 'validation': validation documents
# Labels: EUROVOC concepts at 3 levels (level_1, level_2, level_3)
```

**Dataset Characteristics:**
- **Size:** 65k documents (English subset available)
- **Labels:** Multi-label classification (EUROVOC taxonomy)
- **Average length:** ~727 words per document
- **Levels:** 3 granularities of labels
- **Languages:** 23 languages (use 'en' for English-only)

**For Your Use Case:**
- Small enough for single H100 ($20 budget)
- Complex enough to test FP8 training convergence
- Multi-label classification is sensitive to precision degradation

**Alternative dataset:** EURLEX57K (57k docs, English only) via Papers with Code

**Confidence:** HIGH (official Hugging Face dataset, actively maintained)

---

## Supporting Libraries

### Model Architecture (NanoGPT Base)

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| **nanoGPT** | Latest (Karpathy) | Baseline transformer | Simple, hackable, proven FP8 compatibility |
| **modded-nanoGPT** | Latest (KellerJordan) | FP8 reference implementation | Demonstrates custom FP8 ops on H100 |

**Installation:**
```bash
# Clone nanoGPT
git clone https://github.com/karpathy/nanoGPT.git

# Or modded-nanoGPT for FP8 reference
git clone https://github.com/KellerJordan/modded-nanogpt.git
```

**Why nanoGPT:**
- Simple, readable codebase (~500 lines)
- Easy to modify for custom FP8 formats
- Proven H100 FP8 Tensor Core utilization (modded-nanoGPT)
- Scales to your target 50M params

**modded-nanoGPT insights:**
- Custom `mm_op` for FP8 matmul (E4M3 forward, E5M2 backward)
- Uses `torch.library.custom_op` and `torch._scaled_mm`
- Achieves sub-3-minute training on 8xH100 (GPT-2 124M)

**Confidence:** HIGH (widely used, proven FP8 implementation)

---

### Utilities

| Library | Version | Purpose |
|---------|---------|---------|
| **NumPy** | Latest | Array operations, FP8 bit manipulation |
| **Matplotlib / Seaborn** | Latest | Visualizing FP8 distributions, loss curves |
| **Weights & Biases** | Latest | Experiment tracking (optional) |
| **pytest** | Latest | Testing FP8 quantization correctness |

---

## Installation Script

```bash
#!/bin/bash
# Complete stack installation for FP8 test bench

# Core framework
pip install torch>=2.8 --index-url https://download.pytorch.org/whl/cu124

# FP8 quantization
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu124

# Manifold optimization
pip install git+https://github.com/tfjgeorge/nngeometry.git
pip install git+https://github.com/geoopt/geoopt.git

# Dataset
pip install datasets transformers

# Utilities
pip install numpy matplotlib seaborn wandb pytest

# Optional: NVIDIA Transformer Engine
# pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Optional: QPyTorch for prototyping (NOT for production)
# pip install git+https://github.com/Tiiiger/QPyTorch.git
```

---

## What NOT to Use

### Avoid: Outdated/Incompatible Libraries

| Library | Why Avoid |
|---------|-----------|
| **QPyTorch for production training** | Software simulation only, no H100 acceleration, unmaintained (2019) |
| **TensorFlow/JAX** | Ecosystem less mature for custom FP8 ops; PyTorch has better H100 FP8 support |
| **Transformer Engine for custom formats** | Only supports E4M3/E5M2; not flexible enough for E1M6, E3M4, E7M0 |
| **PyTorch < 2.2** | Missing native float8 types |
| **CUDA < 12.6** | Suboptimal FP8 Tensor Core support |
| **A100/V100 GPUs** | Lack native FP8 Tensor Cores (require H100/Ada/Blackwell) |

### Avoid: Over-Engineering

| Pattern | Why Avoid | What to Do Instead |
|---------|-----------|-------------------|
| **Implementing CUDA kernels from scratch** | Massive time investment, error-prone | Use `torch._scaled_mm` for E4M3/E5M2; emulate exotic formats in PyTorch first |
| **Full MXFP8 implementation** | Block-scaled quantization adds complexity; hardware support only on Blackwell | Start with tensorwise scaling, add rowwise if needed |
| **Distributed training** | Single H100 is sufficient, adds complexity | Use single-GPU training, measure wall-clock time |
| **16-bit baselines (FP16)** | BF16 is strictly better on H100 | Always compare FP8 vs BF16, not FP16 |

---

## Confidence Levels

| Category | Confidence | Evidence |
|----------|------------|----------|
| **PyTorch 2.8 + FP8 native types** | HIGH | Official PyTorch 2.8 release notes, NVIDIA documentation |
| **TorchAO for E4M3/E5M2 training** | HIGH | Production deployments at 2k GPU scale, official PyTorch blog |
| **Custom ops for exotic formats** | MEDIUM | modded-nanoGPT proves viability for E4M3/E5M2; E1M6/E3M4/E7M0 will need custom quantize/dequantize |
| **NNGeometry for Fisher computation** | HIGH | Active development, v0.4 released Apr 2025 |
| **Geoopt for manifold optimization** | HIGH | Mature library, PyTorch 2.0+ compatibility |
| **Custom FP8Manifold implementation** | LOW | No existing library; you'll implement from scratch based on manifold theory |
| **H100 FP8 Tensor Core utilization** | HIGH | Extensively documented by NVIDIA, proven in modded-nanoGPT |
| **EurLex dataset suitability** | HIGH | Well-documented on Hugging Face, appropriate scale for budget |

---

## Hardware-Specific Notes: H100 FP8 Capabilities

Your H100 GPU has the following FP8 characteristics:

| Feature | Specification |
|---------|---------------|
| **FP8 formats** | E4M3, E5M2 (hardware-accelerated) |
| **FP8 TFLOPS** | 3.9 PFLOPS (2x BF16's 1.9 PFLOPS) |
| **Practical speedup** | 1.5-2x over BF16 (with good scaling) |
| **Block-scaled FP8** | Not hardware-accelerated (Blackwell feature) |
| **Tensor Core accumulation** | FP32 or FP16 accumulators |

**Critical limitation:** Exotic formats (E1M6, E3M4, E7M0) will NOT have hardware acceleration.
- **Workaround:** Emulate via INT8 storage + software quantize/dequantize
- **Performance impact:** Expect 2-10x slowdown vs native E4M3/E5M2
- **Research value:** Still valid for studying convergence properties

---

## Recommended Implementation Path

### Phase 1: Baseline with Standard FP8 (Week 1)
1. Set up nanoGPT with TorchAO FP8 (E4M3/E5M2)
2. Train on EurLex with BF16 (baseline)
3. Train on EurLex with FP8 (TorchAO)
4. Measure convergence, wall-clock time, memory

**Stack:** PyTorch 2.8 + TorchAO + nanoGPT + EurLex

### Phase 2: Custom FP8 Operations (Week 2-3)
1. Implement custom FP8 quantize/dequantize for E3M4
2. Replace TorchAO with custom `@custom_op` matmul
3. Verify numerical correctness vs TorchAO
4. Measure performance degradation

**Stack:** PyTorch 2.8 custom_op + modded-nanoGPT reference

### Phase 3: Manifold-Aware Optimizer (Week 4-5)
1. Implement FP8Manifold class in Geoopt
2. Compute stiffness metric (bit-space density)
3. Integrate with RiemannianSGD
4. Compare vs standard SGD on FP8-quantized models

**Stack:** Geoopt + NNGeometry + custom FP8Manifold

### Phase 4: Exotic Formats (Week 6+)
1. Extend to E1M6, E5M2, E7M0
2. Study manifold curvature differences
3. Measure convergence vs standard formats

**Stack:** All of the above + extensive custom code

---

## Open Questions & Risks

### Technical Risks

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| **Exotic formats too slow without CUDA kernels** | HIGH | Start with PyTorch emulation; only write CUDA if performance critical |
| **FP8 manifold theory not well-developed** | MEDIUM | Leverage differential geometry literature; consult with advisors |
| **EurLex too easy/hard for FP8** | LOW | Can swap dataset; MultiEURLEX has multiple difficulty levels |
| **$20 budget insufficient for full experiments** | MEDIUM | Optimize training efficiency; use smaller models if needed |

### Open Research Questions

1. **Does FP8 manifold structure significantly differ across E*M* formats?**
   - Hypothesis: E7M0 (logarithmic) vs E1M6 (near-linear) have vastly different curvature

2. **Is stiffness-based preconditioning better than standard SGD?**
   - Hypothesis: Yes, especially for formats with non-uniform density (E4M3, E5M2)

3. **Can we achieve BF16-equivalent convergence with manifold-aware optimization?**
   - Hypothesis: Manifold-aware methods recover some precision lost to quantization

---

## Sources

### PyTorch FP8 Support
- [PyTorch 2.8 Release Blog](https://pytorch.org/blog/pytorch-2-8/)
- [PyTorch Native FP8 Data Types - Medium](https://medium.com/data-science/pytorch-native-fp8-fedc06f1c9f7)
- [Accelerating 2K Scale Pre-Training with TorchAO - PyTorch Blog](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)

### FP8 Libraries
- [TorchAO GitHub](https://github.com/pytorch/ao)
- [NVIDIA Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [NVIDIA Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [QPyTorch GitHub](https://github.com/Tiiiger/QPyTorch)

### Manifold Optimization
- [Geoopt GitHub](https://github.com/geoopt/geoopt)
- [Geoopt Documentation](https://geoopt.readthedocs.io/)
- [NNGeometry GitHub](https://github.com/tfjgeorge/nngeometry)
- [NNGeometry Documentation](https://nngeometry.readthedocs.io/)
- [wiseodd/natural-gradients GitHub](https://github.com/wiseodd/natural-gradients)

### NanoGPT FP8 Implementations
- [modded-nanoGPT GitHub](https://github.com/KellerJordan/modded-nanogpt)
- [Modded-NanoGPT Walkthrough - Custom FP8 Operations](https://damek.github.io/random/modded-nanogpt-walkthrough-i/)
- [nanoGPT FP8 GitHub](https://github.com/cchan/nanoGPT-fp8)

### H100 FP8 Capabilities
- [NVIDIA Hopper: H100 and FP8 Support - Lambda AI](https://lambda.ai/blog/nvidia-hopper-h100-and-fp8-support)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Floating-Point 8: Introduction to FP8 Training](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)

### Dataset
- [MultiEURLEX - Hugging Face](https://huggingface.co/datasets/nlpaueb/multi_eurlex)
- [MultiEURLEX - ACL Anthology](https://aclanthology.org/2021.emnlp-main.559/)
- [MultiEURLEX - GitHub](https://github.com/nlpaueb/multi-eurlex)
- [EURLEX57K - Papers with Code](https://paperswithcode.com/dataset/eurlex57k)

### Quantization Techniques
- [Neural Network Quantization in PyTorch](https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/)
- [PyTorch Quantization Introduction](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [Low-Bit Precision Training in PyTorch - Medium](https://medium.com/the-owl/low-bit-precision-training-in-pytorch-techniques-and-code-examples-038902ceaaf9)

---

**Final Recommendation:** Start with PyTorch 2.8 + TorchAO for baseline FP8 experiments (E4M3/E5M2), then implement custom operations using `torch.library.custom_op` for exotic formats. Use Geoopt for manifold-aware optimizer experiments, implementing a custom FP8Manifold class. NNGeometry provides Fisher information computation. NanoGPT (or modded-nanoGPT) offers a proven starting point for H100 FP8 training.
