"""FP32 shadow model for gradient comparison.

Maintains a full-precision copy of the model to compare gradient quality
between FP8 quantized training and FP32 baseline. Essential for measuring
the impact of quantization on gradient signal.

Key features:
  - Deep copy model to FP32
  - Sync weights from quantized model
  - Compute gradient similarity (cosine) and SNR comparison
  - Per-layer and aggregate metrics

Example:
    >>> from altgrad.training.shadow import FP32ShadowModel
    >>> from altgrad.training.model import GPT, GPTConfig
    >>> model = GPT(GPTConfig())
    >>> shadow = FP32ShadowModel(model)
    >>> # After forward/backward on both models...
    >>> metrics = shadow.compute_gradient_similarity(model)
    >>> print(f"Mean cosine sim: {metrics['grad_cos_sim/mean']:.4f}")
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from altgrad.training.metrics import gradient_cosine_similarity


def _compute_snr(grad: Tensor) -> float:
    """Compute signal-to-noise ratio for gradient tensor.

    SNR = |mean| / std measures how much the gradient is signal vs noise.
    Higher SNR indicates cleaner gradient with consistent direction.

    Args:
        grad: Gradient tensor

    Returns:
        SNR value, or inf if std is near zero
    """
    flat = grad.flatten().float()
    mean_abs = flat.abs().mean().item()
    std = flat.std().item()
    return mean_abs / std if std > 1e-8 else float("inf")


class FP32ShadowModel:
    """FP32 shadow model for gradient comparison.

    Maintains a full-precision copy of a model to compare gradient quality
    between quantized and FP32 training. Computes both cosine similarity
    (direction alignment) and SNR (signal quality) metrics.

    Attributes:
        model: The FP32 shadow copy

    Example:
        >>> model = GPT(config)
        >>> shadow = FP32ShadowModel(model)
        >>> # Run forward/backward on both
        >>> loss_main = model(x, y)[1]
        >>> loss_main.backward()
        >>> shadow.forward_backward(x, y)
        >>> metrics = shadow.compute_gradient_similarity(model)
    """

    def __init__(self, model: nn.Module):
        """Initialize FP32 shadow model.

        Creates a deep copy of the model converted to FP32 with
        gradients enabled.

        Args:
            model: Source model to shadow
        """
        # Deep copy to FP32
        self.model = copy.deepcopy(model)
        self.model.float()  # Ensure FP32
        self.model.train()

        # Enable gradients on all parameters
        for param in self.model.parameters():
            param.requires_grad_(True)

    def sync_weights(self, source_model: nn.Module) -> None:
        """Synchronize weights from source model.

        Copies weights from source (possibly quantized/mixed precision)
        to shadow model in FP32.

        Args:
            source_model: Model to copy weights from
        """
        with torch.no_grad():
            for (name, shadow_param), (_, source_param) in zip(
                self.model.named_parameters(),
                source_model.named_parameters(),
            ):
                shadow_param.copy_(source_param.float())

    def forward_backward(self, x: Tensor, y: Tensor) -> Tensor:
        """Run forward and backward pass on shadow model.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Loss value from forward pass
        """
        # Zero gradients
        self.model.zero_grad()

        # Forward pass
        logits, loss = self.model(x, y)

        # Backward pass
        if loss is not None:
            loss.backward()

        return loss

    def compute_gradient_similarity(
        self,
        quantized_model: nn.Module,
    ) -> Dict[str, float]:
        """Compute gradient similarity and SNR comparison.

        Compares gradients between the quantized model and this FP32 shadow.
        Returns both cosine similarity (direction) and SNR (signal quality).

        Args:
            quantized_model: Model with gradients from quantized training

        Returns:
            Dictionary with:
              - grad_cos_sim/{name}: Per-layer cosine similarity
              - grad_cos_sim/mean: Mean cosine similarity
              - grad_cos_sim/min: Minimum cosine similarity
              - grad_snr/{name}_fp8: Per-layer SNR for quantized model
              - grad_snr/{name}_fp32: Per-layer SNR for shadow model
              - grad_snr/{name}_diff: SNR difference (positive = FP8 noisier)
              - grad_snr/mean_fp8: Mean SNR for quantized model
              - grad_snr/mean_fp32: Mean SNR for shadow model
              - grad_snr/mean_diff: Mean SNR difference
        """
        # Get cosine similarity from metrics module
        metrics = gradient_cosine_similarity(quantized_model, self.model)

        # Compute per-layer SNR comparison
        snr_fp8_values = []
        snr_fp32_values = []
        snr_diff_values = []

        quant_params = dict(quantized_model.named_parameters())
        shadow_params = dict(self.model.named_parameters())

        for name in quant_params:
            grad_quant = quant_params[name].grad
            grad_shadow = shadow_params[name].grad

            if grad_quant is None or grad_shadow is None:
                continue

            # Compute SNR for each
            snr_fp8 = _compute_snr(grad_quant)
            snr_fp32 = _compute_snr(grad_shadow)

            # SNR difference (positive = FP8 has lower SNR / more noise)
            # Note: We use ratio for interpretability but diff is more robust
            if snr_fp32 != float("inf") and snr_fp8 != float("inf"):
                snr_diff = snr_fp32 - snr_fp8  # Positive if FP8 worse
            else:
                snr_diff = 0.0

            # Store per-layer metrics
            # Sanitize name for W&B logging (replace dots with slashes)
            safe_name = name.replace(".", "/")
            metrics[f"grad_snr/{safe_name}_fp8"] = snr_fp8
            metrics[f"grad_snr/{safe_name}_fp32"] = snr_fp32
            metrics[f"grad_snr/{safe_name}_diff"] = snr_diff

            # Collect for aggregates (skip inf values)
            if snr_fp8 != float("inf"):
                snr_fp8_values.append(snr_fp8)
            if snr_fp32 != float("inf"):
                snr_fp32_values.append(snr_fp32)
            if snr_fp8 != float("inf") and snr_fp32 != float("inf"):
                snr_diff_values.append(snr_diff)

        # Aggregate SNR metrics
        metrics["grad_snr/mean_fp8"] = (
            sum(snr_fp8_values) / len(snr_fp8_values) if snr_fp8_values else 0.0
        )
        metrics["grad_snr/mean_fp32"] = (
            sum(snr_fp32_values) / len(snr_fp32_values) if snr_fp32_values else 0.0
        )
        metrics["grad_snr/mean_diff"] = (
            sum(snr_diff_values) / len(snr_diff_values) if snr_diff_values else 0.0
        )

        return metrics


__all__ = [
    "FP32ShadowModel",
]
