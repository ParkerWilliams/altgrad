"""Model integration for FP8 quantization.

This module provides tools for injecting FP8 quantization into PyTorch models
without modifying their source code. The key components are:

- QuantizedLinear: Wrapper that applies quantization during forward pass
- quantize_model: Surgery function to replace nn.Linear with QuantizedLinear
- dequantize_model: Restore original nn.Linear from wrappers

Example:
    >>> from altgrad.integration import quantize_model, dequantize_model
    >>> from altgrad.quantization import E5M2
    >>> model = MyModel()
    >>> quantize_model(model, E5M2)  # Enable FP8 quantization
    >>> # ... train with quantized weights ...
    >>> dequantize_model(model)  # Restore for inference/export
"""

from altgrad.integration.wrapper import QuantizedLinear
from altgrad.integration.surgery import quantize_model, dequantize_model

__all__ = [
    "QuantizedLinear",
    "quantize_model",
    "dequantize_model",
]
