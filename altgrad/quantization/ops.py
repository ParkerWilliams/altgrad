"""Quantization operations with Straight-Through Estimator gradient.

This module provides autograd-compatible quantize/dequantize operations
that simulate FP8 quantization while maintaining gradient flow via STE.

The Straight-Through Estimator (STE) bypasses non-differentiable quantization:
  Forward: y = quantize(x)  # actual quantization to FP8 levels
  Backward: dx = dy         # gradient passes through unchanged

Simulated Quantization Pattern:
  1. Scale input: x_scaled = x / scale
  2. Quantize: x_q = to_real(to_bits(x_scaled))  # via format's transfer functions
  3. Unscale: x_out = x_q * scale

The "fake quantization" happens in the to_bits/to_real round-trip.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from altgrad.quantization.formats import FP8Format


class QuantizeFunc(torch.autograd.Function):
    """Autograd function for FP8 quantization with STE.

    Forward pass performs actual quantization to FP8 representable values.
    Backward pass passes gradient unchanged (Straight-Through Estimator).
    """

    @staticmethod
    def forward(
        ctx: Any, x: Tensor, format: FP8Format, scale: Tensor
    ) -> Tensor:
        """Quantize tensor to FP8 representable values.

        Args:
            ctx: Autograd context (unused for STE)
            x: Input tensor to quantize
            format: FP8 format specification
            scale: Scale factor for dynamic range

        Returns:
            Tensor with values quantized to FP8 representable levels
        """
        # Scale input for quantization
        x_scaled = x / scale

        # Vectorized quantization
        x_quantized = _vectorized_quantize(x_scaled, format)

        # Unscale output
        return x_quantized * scale

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[Tensor, None, None]:
        """STE backward: pass gradient unchanged.

        Args:
            ctx: Autograd context
            grad_output: Gradient from upstream

        Returns:
            Tuple of (gradient for x, None for format, None for scale)
        """
        # STE: gradient passes through unchanged
        return grad_output, None, None


class DequantizeFunc(torch.autograd.Function):
    """Autograd function for dequantization (scale multiplication).

    Forward pass multiplies by scale factor.
    Backward pass scales gradient by the same factor.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, scale: Tensor) -> Tensor:
        """Dequantize by applying scale multiplication.

        Args:
            ctx: Autograd context
            x: Input tensor (quantized values)
            scale: Scale factor to apply

        Returns:
            Scaled tensor
        """
        ctx.save_for_backward(scale)
        return x * scale

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        """Backward pass: scale gradient by scale factor.

        Args:
            ctx: Autograd context with saved scale
            grad_output: Gradient from upstream

        Returns:
            Tuple of (scaled gradient for x, None for scale)
        """
        (scale,) = ctx.saved_tensors
        # Gradient for x is scaled by the scale factor (chain rule)
        return grad_output * scale, None


def quantize(x: Tensor, format: FP8Format, scale: Tensor) -> Tensor:
    """Quantize tensor to FP8 representable values with STE gradient.

    This function simulates FP8 quantization while storing values in FP32.
    The Straight-Through Estimator passes gradients unchanged during backprop.

    Args:
        x: Input tensor to quantize (any shape)
        format: FP8 format specification (E0M7, E1M6, E3M4, E5M2, or E7M0)
        scale: Scale factor for dynamic range adjustment

    Returns:
        Tensor with values quantized to FP8 representable levels (stored as FP32)

    Example:
        >>> x = torch.randn(10, requires_grad=True)
        >>> y = quantize(x, E5M2, torch.tensor(1.0))
        >>> y.sum().backward()
        >>> x.grad  # All ones due to STE
    """
    return QuantizeFunc.apply(x, format, scale)


def dequantize(x: Tensor, scale: Tensor) -> Tensor:
    """Apply scale multiplication with gradient passthrough.

    Used after quantize() to restore the original scale of values.

    Args:
        x: Input tensor (typically output of quantize)
        scale: Scale factor to apply

    Returns:
        Scaled tensor (x * scale)

    Example:
        >>> q = quantize(x, E5M2, scale)
        >>> y = dequantize(q, scale)  # Approximately reconstructs x
    """
    return DequantizeFunc.apply(x, scale)


def _vectorized_quantize(x: Tensor, format: FP8Format) -> Tensor:
    """Vectorized FP8 quantization without Python loops.

    Converts each element to the nearest FP8 representable value.

    Args:
        x: Scaled input tensor
        format: FP8 format specification

    Returns:
        Tensor with values quantized to FP8 levels
    """
    if format.exponent_bits == 0:
        return _quantize_fixed_point(x, format)
    else:
        return _quantize_floating_point(x, format)


def _quantize_fixed_point(x: Tensor, format: FP8Format) -> Tensor:
    """Quantize to fixed-point format (E0M7).

    E0M7 represents values in [-127/128, 127/128] with uniform spacing.
    """
    # Fixed-point: values in [-max_mantissa/2^M, max_mantissa/2^M]
    max_mantissa = (1 << format.mantissa_bits) - 1
    max_val = max_mantissa / (1 << format.mantissa_bits)

    # Clamp to representable range
    x_clamped = torch.clamp(x, -max_val, max_val)

    # Quantize: scale to mantissa range, round, scale back
    scale_factor = float(1 << format.mantissa_bits)
    x_scaled = x_clamped * scale_factor
    x_rounded = torch.round(x_scaled)
    x_quantized = x_rounded / scale_factor

    return x_quantized


def _quantize_floating_point(x: Tensor, format: FP8Format) -> Tensor:
    """Quantize to floating-point format (E1M6, E3M4, E5M2, E7M0).

    Uses the format's to_bits/to_real for exact FP8 representation.
    Vectorized implementation processes entire tensor without Python loops.
    """
    # Store original shape for final reshape
    original_shape = x.shape

    # Flatten for uniform processing
    x_flat = x.flatten()

    # Get sign
    sign = torch.sign(x_flat)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # Treat 0 as positive
    abs_x = torch.abs(x_flat)

    # Get format parameters
    max_exp_val = (1 << format.exponent_bits) - 1
    max_normal_exp = max_exp_val - 1 if format.has_inf else max_exp_val
    mantissa_scale = float(1 << format.mantissa_bits)
    bias = format.bias

    # Calculate max representable value
    max_representable = format.max_representable_value

    # Smallest denormalized value
    if format.mantissa_bits > 0:
        min_denorm = (2 ** (1 - bias)) / mantissa_scale
    else:
        # E7M0: smallest value is 2^(1-bias)
        min_denorm = 2 ** (1 - bias)

    # Smallest normal value
    min_normal = 2 ** (1 - bias)

    # Initialize output
    output = torch.zeros_like(x_flat)

    # Handle infinity (for formats that support it)
    if format.has_inf:
        inf_mask = torch.isinf(abs_x)
        output = torch.where(inf_mask, sign * float('inf'), output)
    else:
        inf_mask = torch.zeros_like(x_flat, dtype=torch.bool)

    # Handle NaN (for formats that support it)
    nan_mask = torch.isnan(x_flat)
    if format.has_nan:
        output = torch.where(nan_mask, torch.tensor(float('nan')), output)
    else:
        # If no NaN support, treat as zero
        output = torch.where(nan_mask, torch.zeros_like(x_flat), output)

    # Handle overflow (values exceeding max representable)
    overflow_mask = (abs_x > max_representable) & ~inf_mask & ~nan_mask
    if format.has_inf:
        output = torch.where(overflow_mask, sign * float('inf'), output)
    else:
        output = torch.where(overflow_mask, sign * max_representable, output)

    # Handle underflow to zero (values too small to represent)
    underflow_mask = (abs_x < min_denorm / 2) & ~nan_mask
    output = torch.where(underflow_mask, torch.zeros_like(x_flat), output)

    # Mask for values that need normal/denorm quantization
    needs_quant = ~inf_mask & ~nan_mask & ~overflow_mask & ~underflow_mask

    if needs_quant.any():
        # Process normal and denormal values
        abs_x_quant = abs_x[needs_quant]
        sign_quant = sign[needs_quant]

        # Separate denormal and normal values
        is_denorm = abs_x_quant < min_normal

        # Initialize result tensor for quantized values
        quant_result = torch.zeros_like(abs_x_quant)

        # Process denormal values
        if is_denorm.any():
            denorm_vals = abs_x_quant[is_denorm]
            denorm_result = _quantize_denorm_vec(
                denorm_vals, format, bias, mantissa_scale
            )
            quant_result[is_denorm] = denorm_result

        # Process normal values
        is_normal = ~is_denorm
        if is_normal.any():
            normal_vals = abs_x_quant[is_normal]
            normal_result = _quantize_normal_vec(
                normal_vals, format, bias, mantissa_scale, max_normal_exp
            )
            quant_result[is_normal] = normal_result

        # Place results back with proper signs
        output[needs_quant] = sign_quant * quant_result

    # Reshape back to original shape
    return output.reshape(original_shape)


def _quantize_denorm_vec(
    x: Tensor, format: FP8Format, bias: int, mantissa_scale: float
) -> Tensor:
    """Quantize denormalized values (vectorized).

    Denorm: value = 2^(1-bias) * (mantissa / 2^M)
    """
    # mantissa = value * 2^(M + bias - 1)
    scale = 2 ** (format.mantissa_bits + bias - 1)
    mantissa_float = x * scale

    # Round to nearest (we use standard rounding for simplicity)
    mantissa = torch.round(mantissa_float)

    max_mantissa = (1 << format.mantissa_bits) - 1
    mantissa = torch.clamp(mantissa, 0, max_mantissa)

    # Convert back: value = mantissa / 2^(M + bias - 1)
    return mantissa / scale


def _quantize_normal_vec(
    x: Tensor, format: FP8Format, bias: int, mantissa_scale: float, max_normal_exp: int
) -> Tensor:
    """Quantize normalized values (vectorized).

    Normal: value = 2^(exp-bias) * (1 + mantissa / 2^M)
    """
    # Calculate exponent: exp = floor(log2(x)) + bias
    log_x = torch.log2(x)
    exp_raw = torch.floor(log_x) + bias

    # Clamp exponent to valid range
    exponent = torch.clamp(exp_raw, 1, max_normal_exp)

    # Calculate significand: x / 2^(exp-bias)
    significand = x / torch.pow(2.0, exponent - bias)

    # Extract mantissa: (significand - 1) * 2^M
    mantissa_float = (significand - 1.0) * mantissa_scale

    # Round mantissa
    mantissa = torch.round(mantissa_float)

    max_mantissa = (1 << format.mantissa_bits) - 1

    # Handle mantissa overflow (rounds up to next exponent)
    overflow = mantissa > max_mantissa
    mantissa = torch.where(overflow, torch.zeros_like(mantissa), mantissa)
    exponent = torch.where(overflow, exponent + 1, exponent)

    # Handle exponent overflow after mantissa overflow
    exp_overflow = exponent > max_normal_exp
    exponent = torch.where(exp_overflow, torch.tensor(float(max_normal_exp)), exponent)
    mantissa = torch.where(
        exp_overflow, torch.tensor(float(max_mantissa)), mantissa
    )

    # Reconstruct value: 2^(exp-bias) * (1 + mantissa/2^M)
    result = torch.pow(2.0, exponent - bias) * (1.0 + mantissa / mantissa_scale)

    return result


__all__ = [
    "QuantizeFunc",
    "DequantizeFunc",
    "quantize",
    "dequantize",
]
