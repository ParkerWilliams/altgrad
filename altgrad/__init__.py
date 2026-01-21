"""AltGrad: Alternative gradient quantization research library.

Research framework for exploring 8-bit floating-point formats and
geometry-aware optimization algorithms. Provides FP8 quantization,
dynamic scaling, and diagnostic tools for training neural networks
with reduced precision.

Subpackages:
    quantization: FP8 formats, quantization operations, scaling, diagnostics

Example:
    >>> import altgrad
    >>> from altgrad.quantization import E5M2, quantize
    >>> import torch
    >>> x = torch.randn(10)
    >>> y = quantize(x, E5M2, torch.tensor(1.0))
"""

__version__ = "0.1.0"

# Import quantization subpackage
from altgrad import quantization

__all__ = [
    "quantization",
    "__version__",
]
