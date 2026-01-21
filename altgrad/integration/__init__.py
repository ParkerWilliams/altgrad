"""Model integration for FP8 quantization.

This module provides tools for injecting FP8 quantization into PyTorch models
without modifying their source code. The key components are:

- QuantizedLinear: Wrapper that applies quantization during forward pass
- quantize_model: Surgery function to replace nn.Linear with QuantizedLinear
- dequantize_model: Restore original nn.Linear from wrappers
- QuantizationConfig: Per-layer precision configuration
- create_mixed_precision_config: Helper for common GPT patterns

Example (single format):
    >>> from altgrad.integration import quantize_model, dequantize_model
    >>> from altgrad.quantization import E5M2
    >>> model = MyModel()
    >>> quantize_model(model, E5M2)  # Enable FP8 quantization
    >>> # ... train with quantized weights ...
    >>> dequantize_model(model)  # Restore for inference/export

Example (mixed precision):
    >>> from altgrad.integration import quantize_model, create_mixed_precision_config
    >>> config = create_mixed_precision_config(attention_format=None, mlp_format='E5M2')
    >>> quantize_model(model, config=config)  # Attention BF16, MLP FP8
"""

from altgrad.integration.wrapper import QuantizedLinear
from altgrad.integration.surgery import quantize_model, dequantize_model
from altgrad.integration.config import (
    LayerPrecisionRule,
    QuantizationConfig,
    create_mixed_precision_config,
)

__all__ = [
    "QuantizedLinear",
    "quantize_model",
    "dequantize_model",
    "LayerPrecisionRule",
    "QuantizationConfig",
    "create_mixed_precision_config",
]
