"""QuantizedLinear wrapper for FP8 quantization at layer boundaries.

This module provides a wrapper that applies FP8 quantization to nn.Linear
forward passes while maintaining gradient flow via STE.

The wrapper:
- Quantizes weights during forward pass (simulated FP8)
- Tracks amax history for dynamic scaling
- Exposes underlying Linear's weight/bias for optimizer compatibility
- Passes gradients unchanged via Straight-Through Estimator

Example:
    >>> from altgrad.integration import QuantizedLinear
    >>> from altgrad.quantization import E5M2
    >>> linear = nn.Linear(768, 768)
    >>> q_linear = QuantizedLinear(linear, E5M2)
    >>> output = q_linear(input_tensor)  # Weights quantized during forward
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from altgrad.quantization import FP8Format, quantize, AmaxHistory, compute_scale


class QuantizedLinear(nn.Module):
    """Wrapper that applies FP8 quantization to Linear forward pass.

    Wraps an existing nn.Linear and applies quantization during forward:
    1. Track weight/input amax for dynamic scaling
    2. Compute scale from amax history
    3. Quantize weights (and optionally inputs) using STE
    4. Perform linear computation with quantized weights

    The Straight-Through Estimator (STE) passes gradients unchanged through
    quantization, enabling training with quantized forward passes.

    Attributes:
        linear: The underlying nn.Linear module
        format: FP8 format specification (E5M2, E3M4, etc.)
        weight_history: AmaxHistory for weight scaling
        input_history: AmaxHistory for input scaling (optional future use)
        quantize_input: Whether to also quantize inputs (default False)

    Example:
        >>> linear = nn.Linear(10, 5)
        >>> q_linear = QuantizedLinear(linear, E5M2)
        >>> x = torch.randn(4, 10, requires_grad=True)
        >>> y = q_linear(x)
        >>> y.sum().backward()
        >>> assert linear.weight.grad is not None  # Gradient flows via STE
    """

    def __init__(
        self,
        linear: nn.Linear,
        format: FP8Format,
        quantize_input: bool = False,
        history_len: int = 16,
    ):
        """Initialize QuantizedLinear wrapper.

        Args:
            linear: The nn.Linear module to wrap
            format: FP8 format for quantization (E5M2, E3M4, etc.)
            quantize_input: Whether to quantize inputs as well as weights
            history_len: Number of amax values to track for stable scaling
        """
        super().__init__()
        self.linear = linear
        self.format = format
        self.quantize_input = quantize_input

        # Amax history for dynamic scaling
        self.weight_history = AmaxHistory(history_len)
        self.input_history = AmaxHistory(history_len)

    @property
    def weight(self) -> Tensor:
        """Expose underlying Linear's weight for optimizer compatibility."""
        return self.linear.weight

    @property
    def bias(self) -> Optional[Tensor]:
        """Expose underlying Linear's bias (may be None)."""
        return self.linear.bias

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with quantized weights.

        Applies FP8 quantization to weights during forward pass:
        1. Update amax history for current weights
        2. Compute scale from history
        3. Quantize weights via STE
        4. Compute linear with quantized weights

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        # Track weight amax for stable scaling
        self.weight_history.update(self.linear.weight.detach())

        # Compute scale for weights
        weight_amax = self.weight_history.get_amax()
        weight_scale = compute_scale(weight_amax, self.format)
        weight_scale_tensor = torch.tensor(
            weight_scale, device=self.linear.weight.device, dtype=self.linear.weight.dtype
        )

        # Quantize weights via STE
        quantized_weight = quantize(self.linear.weight, self.format, weight_scale_tensor)

        # Optionally quantize input
        if self.quantize_input:
            self.input_history.update(x.detach())
            input_amax = self.input_history.get_amax()
            input_scale = compute_scale(input_amax, self.format)
            input_scale_tensor = torch.tensor(
                input_scale, device=x.device, dtype=x.dtype
            )
            x = quantize(x, self.format, input_scale_tensor)

        # Compute linear with quantized weights
        # Use F.linear to use quantized weight while keeping gradient flow
        output = torch.nn.functional.linear(x, quantized_weight, self.linear.bias)

        return output

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        return (
            f"in_features={self.linear.in_features}, "
            f"out_features={self.linear.out_features}, "
            f"format={self.format.name}, "
            f"quantize_input={self.quantize_input}"
        )


__all__ = ["QuantizedLinear"]
