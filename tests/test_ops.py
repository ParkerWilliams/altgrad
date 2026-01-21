"""Tests for quantization operations with Straight-Through Estimator gradient.

These tests validate:
1. Quantize converts FP32 tensors to simulated FP8 representation
2. Dequantize converts simulated FP8 back to FP32
3. STE passes gradients through quantize/dequantize unchanged
4. Quantization respects format's representable range
"""

import math

import pytest
import torch

from altgrad.quantization import E0M7, E1M6, E3M4, E5M2, E7M0
from altgrad.quantization.ops import (
    QuantizeFunc,
    DequantizeFunc,
    quantize,
    dequantize,
)


class TestQuantizeBasic:
    """Basic quantize functionality tests."""

    def test_quantize_returns_tensor(self) -> None:
        """quantize() returns a tensor of the same shape."""
        x = torch.randn(10, 20)
        scale = torch.tensor(1.0)
        y = quantize(x, E5M2, scale)

        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape
        assert y.dtype == torch.float32  # Simulated quantization stores in FP32

    def test_quantize_values_are_fp8_representable(self) -> None:
        """Output values match format's discrete levels exactly."""
        # E5M2 has specific representable values
        # For values near 1.0: 0.875, 1.0, 1.25, 1.5, 1.75, 2.0...
        x = torch.tensor([1.0, 1.1, 1.3, 1.6, 2.0])
        scale = torch.tensor(1.0)
        y = quantize(x, E5M2, scale)

        # Each output should be exactly representable in E5M2
        for val in y.tolist():
            # to_bits then to_real should give back exact same value
            bits = E5M2.to_bits(val)
            reconstructed = E5M2.to_real(bits)
            assert abs(val - reconstructed) < 1e-7, f"{val} not representable in E5M2"

    def test_quantize_respects_scale(self) -> None:
        """scale parameter affects quantization range correctly."""
        x = torch.tensor([2.0, 4.0, 6.0])

        # With scale=2.0, values are treated as if they were half
        y_scaled = quantize(x, E5M2, torch.tensor(2.0))

        # With scale=1.0, same input
        y_unscaled = quantize(x / 2, E5M2, torch.tensor(1.0))

        # The outputs should be equivalent after accounting for scale
        # y_scaled should be roughly 2x y_unscaled
        assert torch.allclose(y_scaled / 2, y_unscaled, atol=1e-5)


class TestStraightThroughEstimator:
    """Tests for STE gradient behavior."""

    def test_ste_gradient_passthrough(self) -> None:
        """STE passes gradient unchanged through quantization."""
        x = torch.randn(10, requires_grad=True)
        scale = torch.tensor(1.0)

        y = quantize(x, E5M2, scale)
        loss = y.sum()
        loss.backward()

        # Gradient should be all ones (d(sum)/dx = 1 for each element)
        expected_grad = torch.ones_like(x)
        assert x.grad is not None
        assert torch.allclose(x.grad, expected_grad), (
            f"Gradient mismatch: got {x.grad}, expected {expected_grad}"
        )

    def test_gradient_shape_preserved(self) -> None:
        """Gradient has same shape as input regardless of quantization."""
        shapes = [(5,), (3, 4), (2, 3, 4), (1, 2, 3, 4)]
        scale = torch.tensor(1.0)

        for shape in shapes:
            x = torch.randn(shape, requires_grad=True)
            y = quantize(x, E5M2, scale)
            y.sum().backward()

            assert x.grad is not None
            assert x.grad.shape == shape, f"Shape {shape}: grad shape {x.grad.shape}"

    def test_ste_with_complex_loss(self) -> None:
        """STE works with more complex loss functions."""
        x = torch.randn(5, 5, requires_grad=True)
        scale = torch.tensor(1.0)

        y = quantize(x, E5M2, scale)
        loss = (y ** 2).mean()  # MSE-like loss
        loss.backward()

        # Gradient should be 2*y/N (derivative of mean squared)
        # But STE means d(quantize)/dx = 1, so gradient flows through
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Gradient should be 2*y/25 for 5x5 tensor
        expected = 2 * y.detach() / 25
        assert torch.allclose(x.grad, expected, atol=1e-5)


class TestQuantizeEdgeCases:
    """Tests for edge cases in quantization."""

    def test_quantize_clamps_out_of_range(self) -> None:
        """Values exceeding format max clamp to max representable value."""
        large_value = torch.tensor([1e10])
        scale = torch.tensor(1.0)

        y = quantize(large_value, E5M2, scale)

        # E5M2 max (with inf/nan support) is around 57344
        max_val = E5M2.max_representable_value
        assert y.item() <= max_val or math.isinf(y.item())

    def test_quantize_handles_zeros(self) -> None:
        """Zero values are preserved through quantization."""
        x = torch.tensor([0.0, -0.0])
        scale = torch.tensor(1.0)

        y = quantize(x, E5M2, scale)

        assert y[0].item() == 0.0
        assert y[1].item() == 0.0 or y[1].item() == -0.0

    def test_quantize_handles_negative_values(self) -> None:
        """Negative values are quantized correctly."""
        x = torch.tensor([-1.0, -2.5, -0.5])
        scale = torch.tensor(1.0)

        y = quantize(x, E5M2, scale)

        # All outputs should be negative
        assert (y <= 0).all()
        # Should be representable
        for val in y.tolist():
            bits = E5M2.to_bits(val)
            reconstructed = E5M2.to_real(bits)
            assert abs(val - reconstructed) < 1e-7

    def test_quantize_handles_small_values(self) -> None:
        """Very small values are handled (either quantized or zeroed)."""
        # E5M2 smallest denorm is around 2^-16
        tiny = torch.tensor([1e-10, 1e-6, 1e-4])
        scale = torch.tensor(1.0)

        y = quantize(tiny, E5M2, scale)

        # Should not error, values should be finite
        assert torch.isfinite(y).all()


class TestAllFormats:
    """Tests that quantize works with all 5 FP8 formats."""

    @pytest.mark.parametrize("fmt", [E0M7, E1M6, E3M4, E5M2, E7M0])
    def test_quantize_all_formats(self, fmt) -> None:
        """quantize works with all FP8 format specifications."""
        x = torch.tensor([0.5, 0.25, 0.125])
        scale = torch.tensor(1.0)

        y = quantize(x, fmt, scale)

        assert y.shape == x.shape
        assert y.dtype == torch.float32

        # Verify outputs are representable in the format
        for val in y.tolist():
            bits = fmt.to_bits(val)
            reconstructed = fmt.to_real(bits)
            assert abs(val - reconstructed) < 1e-6, (
                f"Format {fmt.name}: {val} not representable"
            )

    @pytest.mark.parametrize("fmt", [E0M7, E1M6, E3M4, E5M2, E7M0])
    def test_gradient_all_formats(self, fmt) -> None:
        """Gradient flows correctly through all formats."""
        x = torch.randn(10, requires_grad=True)
        scale = torch.tensor(1.0)

        y = quantize(x, fmt, scale)
        y.sum().backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))


class TestDequantize:
    """Tests for dequantize operation."""

    def test_dequantize_with_scale(self) -> None:
        """dequantize applies scale multiplication correctly."""
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(2.0)

        y = dequantize(x, scale)

        expected = x * scale
        assert torch.allclose(y, expected)

    def test_dequantize_gradient(self) -> None:
        """dequantize passes gradient through (STE behavior)."""
        x = torch.randn(10, requires_grad=True)
        scale = torch.tensor(2.0)

        y = dequantize(x, scale)
        y.sum().backward()

        # Gradient should scale by the scale factor
        expected_grad = torch.ones_like(x) * scale
        assert x.grad is not None
        assert torch.allclose(x.grad, expected_grad)


class TestRoundtripGradient:
    """Tests for gradient flow through quantize -> dequantize chain."""

    def test_roundtrip_gradient_flow(self) -> None:
        """Gradient flows through entire quantize -> dequantize chain."""
        x = torch.randn(10, requires_grad=True)
        scale = torch.tensor(1.0)

        # Forward: x -> quantize -> dequantize -> loss
        q = quantize(x, E5M2, scale)
        y = dequantize(q, scale)
        loss = y.sum()
        loss.backward()

        # With scale=1 and STE, gradient should be ones
        expected_grad = torch.ones_like(x)
        assert x.grad is not None
        assert torch.allclose(x.grad, expected_grad)

    def test_roundtrip_with_scale(self) -> None:
        """Gradient flows correctly through chain with non-unit scale."""
        x = torch.randn(10, requires_grad=True)
        scale = torch.tensor(2.0)

        q = quantize(x, E5M2, scale)
        y = dequantize(q, scale)
        loss = y.sum()
        loss.backward()

        # quantize passes gradient unchanged, dequantize multiplies by scale
        expected_grad = torch.ones_like(x) * scale
        assert x.grad is not None
        assert torch.allclose(x.grad, expected_grad)

    def test_roundtrip_preserves_approximate_values(self) -> None:
        """Quantize -> dequantize approximately preserves values."""
        x = torch.tensor([0.5, 1.0, 1.5, 2.0])
        scale = torch.tensor(1.0)

        q = quantize(x, E5M2, scale)
        y = dequantize(q, scale)

        # Values should be close (within quantization error)
        # E5M2 has 2 mantissa bits, so ~25% relative error at worst
        assert torch.allclose(x, y, rtol=0.3)


class TestAutograd:
    """Tests for autograd.Function implementation details."""

    def test_quantize_func_is_autograd_function(self) -> None:
        """QuantizeFunc inherits from torch.autograd.Function."""
        assert hasattr(QuantizeFunc, 'apply')
        assert hasattr(QuantizeFunc, 'forward')
        assert hasattr(QuantizeFunc, 'backward')

    def test_dequantize_func_is_autograd_function(self) -> None:
        """DequantizeFunc inherits from torch.autograd.Function."""
        assert hasattr(DequantizeFunc, 'apply')
        assert hasattr(DequantizeFunc, 'forward')
        assert hasattr(DequantizeFunc, 'backward')

    def test_gradient_does_not_propagate_to_scale(self) -> None:
        """Scale parameter does not receive gradient."""
        x = torch.randn(10, requires_grad=True)
        scale = torch.tensor(1.0, requires_grad=True)

        y = quantize(x, E5M2, scale)
        y.sum().backward()

        # x should have gradient
        assert x.grad is not None

        # scale should not have gradient (passed None in backward)
        assert scale.grad is None


class TestVectorization:
    """Tests verifying vectorized implementation (no Python loops)."""

    def test_large_tensor_performance(self) -> None:
        """Large tensors are quantized efficiently (vectorized)."""
        import time

        x = torch.randn(1000, 1000)
        scale = torch.tensor(1.0)

        start = time.time()
        y = quantize(x, E5M2, scale)
        elapsed = time.time() - start

        # Should complete in reasonable time (<1s for 1M elements)
        # A non-vectorized implementation would be much slower
        assert elapsed < 1.0, f"Quantization took {elapsed:.2f}s, expected <1s"
        assert y.shape == x.shape

    def test_batch_dimension_preserved(self) -> None:
        """Batch dimensions are handled correctly."""
        x = torch.randn(4, 8, 16, 32)  # Batch of images
        scale = torch.tensor(1.0)

        y = quantize(x, E5M2, scale)

        assert y.shape == x.shape
