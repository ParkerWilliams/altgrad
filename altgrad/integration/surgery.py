"""Model surgery functions for FP8 quantization injection.

This module provides functions to modify PyTorch models by replacing
nn.Linear layers with QuantizedLinear wrappers, and vice versa.

The surgery approach enables FP8 quantization without modifying model
source code - quantization is injected post-construction.

Key functions:
- quantize_model: Replace nn.Linear with QuantizedLinear
- dequantize_model: Restore nn.Linear from QuantizedLinear wrappers

Two modes of operation:
1. Single format: Apply same FP8 format to all layers (except skip_patterns)
2. Config-based: Apply per-layer formats based on QuantizationConfig rules

Example:
    >>> from altgrad.integration import quantize_model, dequantize_model
    >>> from altgrad.quantization import E5M2
    >>> model = GPT(config)
    >>> quantize_model(model, E5M2, skip_patterns=['lm_head'])
    >>> # ... train with quantized weights ...
    >>> dequantize_model(model)  # Restore for export

    # Or with per-layer precision:
    >>> from altgrad.integration import create_mixed_precision_config
    >>> config = create_mixed_precision_config(attention_format=None, mlp_format='E5M2')
    >>> quantize_model(model, config=config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch.nn as nn

from altgrad.quantization import FP8Format
from altgrad.integration.wrapper import QuantizedLinear

if TYPE_CHECKING:
    from altgrad.integration.config import QuantizationConfig


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by its full dotted name.

    Args:
        model: Root module
        name: Dotted path like "transformer.h.0.attn.c_attn"

    Returns:
        The submodule at the specified path
    """
    parts = name.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a submodule by its full dotted name.

    Args:
        model: Root module
        name: Dotted path like "transformer.h.0.attn.c_attn"
        new_module: Module to place at that path
    """
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _should_skip(name: str, skip_patterns: List[str]) -> bool:
    """Check if a module name matches any skip pattern.

    Args:
        name: Full module name (dotted path)
        skip_patterns: List of patterns to skip

    Returns:
        True if module should be skipped
    """
    for pattern in skip_patterns:
        if pattern in name:
            return True
    return False


def quantize_model(
    model: nn.Module,
    format: Optional[FP8Format] = None,
    skip_patterns: Optional[List[str]] = None,
    quantize_input: bool = False,
    history_len: int = 16,
    config: Optional["QuantizationConfig"] = None,
) -> None:
    """Replace nn.Linear modules with QuantizedLinear wrappers.

    Performs in-place surgery on the model, replacing nn.Linear modules
    with QuantizedLinear wrappers that apply FP8 quantization during
    forward passes.

    Two modes of operation:
    1. Single format mode: Provide `format` to apply same FP8 format to all
       layers (except those matching skip_patterns)
    2. Config mode: Provide `config` for per-layer format selection based on
       regex pattern matching

    IMPORTANT: For models with weight tying (e.g., GPT's wte/lm_head),
    use skip_patterns (single format mode) or set format=None in config
    rules to preserve the weight sharing.

    Args:
        model: The model to quantize (modified in-place)
        format: FP8 format for all layers (mutually exclusive with config)
        skip_patterns: Layer name patterns to skip (only used with format)
        quantize_input: Whether to also quantize layer inputs (only with format)
        history_len: Amax history length for dynamic scaling (only with format)
        config: QuantizationConfig for per-layer precision (mutually exclusive with format)

    Raises:
        ValueError: If neither or both format and config are provided

    Example:
        >>> # Single format mode
        >>> from altgrad.integration import quantize_model
        >>> from altgrad.quantization import E5M2
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        >>> quantize_model(model, E5M2)
        >>> isinstance(model[0], QuantizedLinear)  # True

        >>> # Config mode (per-layer precision)
        >>> from altgrad.integration import create_mixed_precision_config
        >>> config = create_mixed_precision_config(attention_format=None, mlp_format='E5M2')
        >>> quantize_model(model, config=config)
    """
    # Import here to avoid circular import
    from altgrad.integration.config import QuantizationConfig

    # Validate arguments: exactly one of format or config must be provided
    if (format is None) == (config is None):
        raise ValueError(
            "Provide exactly one of 'format' or 'config'. "
            "Use format=E5M2 for single format, or config=QuantizationConfig(...) for per-layer precision."
        )

    if skip_patterns is None:
        skip_patterns = []

    # Collect modules to replace (don't modify while iterating)
    modules_to_replace: List[tuple] = []

    for name, module in model.named_modules():
        # Skip QuantizedLinear (already wrapped)
        if isinstance(module, QuantizedLinear):
            continue

        # Only replace nn.Linear
        if not isinstance(module, nn.Linear):
            continue

        if config is not None:
            # Config mode: use config to determine format for each layer
            layer_format = config.get_format_for_layer(name)
            if layer_format is None:
                # None means keep in BF16, skip quantization
                continue
            modules_to_replace.append((name, module, layer_format, config.history_len, config.quantize_input))
        else:
            # Single format mode: check skip patterns
            if _should_skip(name, skip_patterns):
                continue
            modules_to_replace.append((name, module, format, history_len, quantize_input))

    # Replace modules
    for name, module, layer_format, hist_len, quant_input in modules_to_replace:
        wrapper = QuantizedLinear(
            module,
            layer_format,
            quantize_input=quant_input,
            history_len=hist_len,
        )
        _set_module_by_name(model, name, wrapper)


def dequantize_model(model: nn.Module) -> None:
    """Restore nn.Linear modules from QuantizedLinear wrappers.

    Performs in-place surgery on the model, replacing QuantizedLinear
    wrappers with their underlying nn.Linear modules.

    Args:
        model: The model to dequantize (modified in-place)

    Example:
        >>> from altgrad.integration import quantize_model, dequantize_model
        >>> model = nn.Sequential(nn.Linear(10, 5))
        >>> quantize_model(model, E5M2)
        >>> isinstance(model[0], QuantizedLinear)  # True
        >>> dequantize_model(model)
        >>> isinstance(model[0], nn.Linear)  # True (not QuantizedLinear)
    """
    # Collect modules to restore (don't modify while iterating)
    modules_to_restore: List[tuple] = []

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            modules_to_restore.append((name, module))

    # Restore modules
    for name, wrapper in modules_to_restore:
        _set_module_by_name(model, name, wrapper.linear)


__all__ = ["quantize_model", "dequantize_model"]
