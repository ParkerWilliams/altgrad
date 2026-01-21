"""Gradient and stability metrics computation.

Provides functions for computing per-layer gradient statistics, stability
metrics, and gradient similarity between models. Essential for monitoring
training health and diagnosing quantization issues.

Key metrics:
  - L2/Linf gradient norms: Detect exploding/vanishing gradients
  - Dead neuron fraction: Identify frozen parameters
  - Signal-to-noise ratio: Assess gradient quality
  - Cosine similarity: Compare gradient directions between runs

Example:
    >>> import torch.nn as nn
    >>> model = nn.Linear(10, 10)
    >>> # After backward pass...
    >>> stats = compute_gradient_stats(model)
    >>> print(stats['grad_norm_l2/mean'])
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from altgrad.quantization.diagnostics import BitStallDetector


def compute_gradient_stats(
    model: nn.Module,
    threshold: float = 1e-8,
) -> Dict[str, float]:
    """Compute per-layer gradient statistics.

    For each parameter with gradients, computes:
      - L2 norm (Frobenius norm)
      - Linf norm (max absolute value)
      - Dead neuron fraction (gradient elements below threshold)
      - Signal-to-noise ratio (|mean| / std)

    Args:
        model: PyTorch model with computed gradients
        threshold: Magnitude below which gradient is considered "dead"

    Returns:
        Dictionary with keys:
          - grad_norm_l2/{name}: L2 norm for each parameter
          - grad_norm_linf/{name}: Linf norm for each parameter
          - dead_neuron_frac/{name}: Fraction of elements < threshold
          - grad_snr/{name}: Signal-to-noise ratio
          - grad_norm_l2/mean: Mean L2 norm across layers
          - grad_norm_l2/min: Minimum L2 norm (vanishing gradient indicator)
          - grad_norm_linf/max: Maximum Linf norm (exploding gradient indicator)
          - dead_neuron_frac/mean: Mean dead fraction across layers
          - grad_snr/mean: Mean SNR across layers
          - grad_snr/min: Minimum SNR across layers

    Example:
        >>> model = nn.Linear(100, 100)
        >>> x = torch.randn(32, 100)
        >>> model(x).sum().backward()
        >>> stats = compute_gradient_stats(model)
        >>> print(f"L2 norm: {stats['grad_norm_l2/mean']:.4f}")
    """
    stats: Dict[str, float] = {}
    l2_norms = []
    linf_norms = []
    dead_fracs = []
    snrs = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()
        flat_grad = grad.flatten().float()

        # L2 norm (Frobenius)
        l2 = torch.norm(flat_grad, p=2).item()
        stats[f"grad_norm_l2/{name}"] = l2
        l2_norms.append(l2)

        # Linf norm (max absolute)
        linf = torch.norm(flat_grad, p=float("inf")).item()
        stats[f"grad_norm_linf/{name}"] = linf
        linf_norms.append(linf)

        # Dead neuron fraction
        dead_mask = flat_grad.abs() < threshold
        dead_frac = dead_mask.float().mean().item()
        stats[f"dead_neuron_frac/{name}"] = dead_frac
        dead_fracs.append(dead_frac)

        # Signal-to-noise ratio: |mean| / std
        mean_val = flat_grad.mean().abs().item()
        std_val = flat_grad.std().item()
        snr = mean_val / std_val if std_val > 1e-10 else 0.0
        stats[f"grad_snr/{name}"] = snr
        snrs.append(snr)

    # Aggregates
    if l2_norms:
        stats["grad_norm_l2/mean"] = sum(l2_norms) / len(l2_norms)
        stats["grad_norm_l2/min"] = min(l2_norms)
    else:
        stats["grad_norm_l2/mean"] = 0.0
        stats["grad_norm_l2/min"] = 0.0

    if linf_norms:
        stats["grad_norm_linf/max"] = max(linf_norms)
    else:
        stats["grad_norm_linf/max"] = 0.0

    if dead_fracs:
        stats["dead_neuron_frac/mean"] = sum(dead_fracs) / len(dead_fracs)
    else:
        stats["dead_neuron_frac/mean"] = 0.0

    if snrs:
        stats["grad_snr/mean"] = sum(snrs) / len(snrs)
        stats["grad_snr/min"] = min(snrs)
    else:
        stats["grad_snr/mean"] = 0.0
        stats["grad_snr/min"] = 0.0

    return stats


def compute_stability_metrics(
    model: nn.Module,
    detector: Optional["BitStallDetector"] = None,
) -> Dict[str, float]:
    """Compute stability metrics for model parameters.

    Checks for numerical issues (NaN, Inf) in parameters and optionally
    includes bit-stall rate from a BitStallDetector.

    Args:
        model: PyTorch model to check
        detector: Optional BitStallDetector for stall rate

    Returns:
        Dictionary with keys:
          - param_nan_count: Total NaN values in parameters
          - param_inf_count: Total Inf values in parameters
          - bit_stall_rate: Stall rate from detector (if provided)

    Example:
        >>> model = nn.Linear(100, 100)
        >>> metrics = compute_stability_metrics(model)
        >>> assert metrics['param_nan_count'] == 0
    """
    nan_count = 0
    inf_count = 0

    for param in model.parameters():
        nan_count += torch.isnan(param.data).sum().item()
        inf_count += torch.isinf(param.data).sum().item()

    metrics = {
        "param_nan_count": nan_count,
        "param_inf_count": inf_count,
    }

    if detector is not None:
        metrics["bit_stall_rate"] = detector.get_stall_rate()

    return metrics


def gradient_cosine_similarity(
    model_a: nn.Module,
    model_b: nn.Module,
) -> Dict[str, float]:
    """Compute per-layer cosine similarity between gradients.

    Useful for comparing gradient directions between:
      - FP32 and FP8 training runs
      - Different FP8 formats
      - Before/after quantization

    Args:
        model_a: First model with computed gradients
        model_b: Second model with computed gradients (same architecture)

    Returns:
        Dictionary with keys:
          - grad_cos_sim/{name}: Cosine similarity for each parameter [-1, 1]
          - grad_cos_sim/mean: Mean similarity across layers
          - grad_cos_sim/min: Minimum similarity (worst alignment)

    Raises:
        ValueError: If models have different architectures

    Example:
        >>> model_fp32 = nn.Linear(100, 100)
        >>> model_fp8 = nn.Linear(100, 100)
        >>> # Run forward/backward on same input...
        >>> sim = gradient_cosine_similarity(model_fp32, model_fp8)
        >>> print(f"Mean similarity: {sim['grad_cos_sim/mean']:.4f}")
    """
    similarities: Dict[str, float] = {}
    sim_values = []

    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    if set(params_a.keys()) != set(params_b.keys()):
        raise ValueError("Models have different parameter names")

    for name in params_a:
        grad_a = params_a[name].grad
        grad_b = params_b[name].grad

        if grad_a is None or grad_b is None:
            continue

        # Flatten and compute cosine similarity
        flat_a = grad_a.detach().flatten().float().unsqueeze(0)
        flat_b = grad_b.detach().flatten().float().unsqueeze(0)

        # F.cosine_similarity returns tensor of shape [1]
        cos_sim = F.cosine_similarity(flat_a, flat_b).item()
        similarities[f"grad_cos_sim/{name}"] = cos_sim
        sim_values.append(cos_sim)

    # Aggregates
    if sim_values:
        similarities["grad_cos_sim/mean"] = sum(sim_values) / len(sim_values)
        similarities["grad_cos_sim/min"] = min(sim_values)
    else:
        similarities["grad_cos_sim/mean"] = 0.0
        similarities["grad_cos_sim/min"] = 0.0

    return similarities


__all__ = [
    "compute_gradient_stats",
    "compute_stability_metrics",
    "gradient_cosine_similarity",
]
