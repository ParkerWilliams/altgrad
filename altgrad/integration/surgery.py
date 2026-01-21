"""Model surgery functions for FP8 quantization injection.

This module provides functions to modify PyTorch models by replacing
nn.Linear layers with QuantizedLinear wrappers, and vice versa.

The surgery approach enables FP8 quantization without modifying model
source code - quantization is injected post-construction.

Key functions:
- quantize_model: Replace nn.Linear with QuantizedLinear
- dequantize_model: Restore nn.Linear from QuantizedLinear wrappers

Example:
    >>> from altgrad.integration import quantize_model, dequantize_model
    >>> from altgrad.quantization import E5M2
    >>> model = GPT(config)
    >>> quantize_model(model, E5M2, skip_patterns=['lm_head'])
    >>> # ... train with quantized weights ...
    >>> dequantize_model(model)  # Restore for export
"""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn

from altgrad.quantization import FP8Format
from altgrad.integration.wrapper import QuantizedLinear


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
    format: FP8Format,
    skip_patterns: Optional[List[str]] = None,
    quantize_input: bool = False,
    history_len: int = 16,
) -> None:
    """Replace nn.Linear modules with QuantizedLinear wrappers.

    Performs in-place surgery on the model, replacing nn.Linear modules
    with QuantizedLinear wrappers that apply FP8 quantization during
    forward passes.

    IMPORTANT: For models with weight tying (e.g., GPT's wte/lm_head),
    skip one of the tied modules to preserve the weight sharing.

    Args:
        model: The model to quantize (modified in-place)
        format: FP8 format for quantization (E5M2, E3M4, etc.)
        skip_patterns: List of name substrings to skip (e.g., ['lm_head'])
        quantize_input: Whether to also quantize layer inputs
        history_len: Amax history length for dynamic scaling

    Example:
        >>> from altgrad.integration import quantize_model
        >>> from altgrad.quantization import E5M2
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        >>> quantize_model(model, E5M2)
        >>> isinstance(model[0], QuantizedLinear)  # True
    """
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

        # Check skip patterns
        if _should_skip(name, skip_patterns):
            continue

        modules_to_replace.append((name, module))

    # Replace modules
    for name, module in modules_to_replace:
        wrapper = QuantizedLinear(
            module,
            format,
            quantize_input=quantize_input,
            history_len=history_len,
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
