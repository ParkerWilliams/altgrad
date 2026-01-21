"""FP8 quantization formats, transfer functions, and operations.

This package provides the mathematical foundation for FP8 quantization,
including format specifications, bit-level transfer functions, and
autograd-compatible quantization operations with STE gradient.

Available Formats:
    E0M7: Fixed-point in [-1, 1), 7 mantissa bits, no exponent
    E1M6: Two-scale format, 1 exponent bit, 6 mantissa bits
    E3M4: Moderate range, 3 exponent bits, 4 mantissa bits
    E5M2: Standard FP8 with wide range, IEEE-like semantics
    E7M0: Powers of 2 only, 7 exponent bits, no mantissa

Operations:
    quantize: Convert FP32 tensor to simulated FP8 with STE gradient
    dequantize: Apply scale multiplication with gradient passthrough

Example:
    >>> from altgrad.quantization import E5M2, quantize
    >>> import torch
    >>> x = torch.randn(10, requires_grad=True)
    >>> y = quantize(x, E5M2, torch.tensor(1.0))
    >>> y.sum().backward()
    >>> x.grad  # All ones due to STE
"""

from altgrad.quantization.formats import (
    FP8Format,
    E0M7,
    E1M6,
    E3M4,
    E5M2,
    E7M0,
    FORMAT_REGISTRY,
)
from altgrad.quantization.ops import (
    QuantizeFunc,
    DequantizeFunc,
    quantize,
    dequantize,
)

__all__ = [
    # Formats
    "FP8Format",
    "E0M7",
    "E1M6",
    "E3M4",
    "E5M2",
    "E7M0",
    "FORMAT_REGISTRY",
    # Operations
    "QuantizeFunc",
    "DequantizeFunc",
    "quantize",
    "dequantize",
]
