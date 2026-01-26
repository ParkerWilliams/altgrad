"""Rank health monitoring for weight matrices.

Provides stable rank, effective rank, and rank collapse early warning
for monitoring weight matrix health during training. These metrics
detect gradual rank degradation before catastrophic failure.

Key Metrics:
  - Stable rank: ||W||_F^2 / ||W||_2^2 = sum(s_i^2) / s_1^2
    Measures "effective dimension" of weight matrix. Robust to noise.

  - Effective rank: exp(-sum(p_i * log(p_i))) where p_i = s_i / sum(s_j)
    Entropy-based measure. More sensitive to singular value distribution.

Why it matters:
  Rank collapse during training indicates:
  - Weights converging to low-rank subspace
  - Loss of representational capacity
  - Potential gradient pathology (updates only in few directions)

Detection approach:
  1. Compute rank metrics per layer at intervals
  2. Track EMA trend after warmup period
  3. Warn if sustained downward trend exceeds threshold

Example:
    >>> import torch
    >>> from altgrad.quantization.rank_health import (
    ...     compute_stable_rank, compute_effective_rank, RankHealthMonitor
    ... )
    >>> w = torch.randn(64, 128)
    >>> print(f"Stable rank: {compute_stable_rank(w):.2f}")
    >>> print(f"Effective rank: {compute_effective_rank(w):.2f}")
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


def compute_stable_rank(weight: Tensor) -> float:
    """Compute stable rank of a weight matrix.

    Stable rank is defined as ||W||_F^2 / ||W||_2^2 = sum(s_i^2) / s_1^2
    where s_i are singular values. It measures the "effective dimension"
    of the matrix and is bounded by [1, min(m, n)].

    Args:
        weight: Weight tensor (2D or higher). Higher-dim tensors are
                reshaped to 2D: (out_features, -1).

    Returns:
        Stable rank as float. Returns weight.numel() for zero matrix.

    Example:
        >>> eye = torch.eye(10)
        >>> compute_stable_rank(eye)  # Returns ~10.0
        >>> rank1 = torch.randn(10, 1) @ torch.randn(1, 5)
        >>> compute_stable_rank(rank1)  # Returns ~1.0
    """
    # Handle scalar or 1D tensors
    if weight.dim() < 2:
        return float(weight.numel())

    # Reshape to 2D for SVD: (out_features, in_features * ...)
    w_2d = weight.view(weight.size(0), -1).float()

    # Compute singular values only (much faster than full SVD)
    try:
        sv = torch.linalg.svdvals(w_2d)
    except RuntimeError:
        # SVD failed (e.g., all zeros, NaN)
        return float(weight.numel())

    # Handle zero matrix: all singular values are zero
    s1_sq = sv[0].item() ** 2
    if s1_sq < 1e-10:
        return float(min(w_2d.size(0), w_2d.size(1)))

    # Stable rank = sum(s_i^2) / s_1^2 = ||W||_F^2 / ||W||_2^2
    frobenius_sq = (sv ** 2).sum().item()
    stable_rank = frobenius_sq / s1_sq

    return stable_rank


def compute_effective_rank(weight: Tensor, eps: float = 1e-10) -> float:
    """Compute effective rank of a weight matrix.

    Effective rank is defined as exp(-sum(p_i * log(p_i))) where
    p_i = s_i / sum(s_j) are normalized singular values. This is the
    exponential of the entropy of the normalized singular value distribution.

    Args:
        weight: Weight tensor (2D or higher). Higher-dim tensors are
                reshaped to 2D: (out_features, -1).
        eps: Small value to avoid log(0) and division by zero.

    Returns:
        Effective rank as float in [1, min(m, n)].
        Returns 1.0 for degenerate matrices.

    Example:
        >>> eye = torch.eye(10)
        >>> compute_effective_rank(eye)  # Returns ~10.0
        >>> rank1 = torch.randn(10, 1) @ torch.randn(1, 5)
        >>> compute_effective_rank(rank1)  # Returns ~1.0
    """
    # Handle scalar or 1D tensors
    if weight.dim() < 2:
        return 1.0

    # Reshape to 2D for SVD
    w_2d = weight.view(weight.size(0), -1).float()

    # Compute singular values only
    try:
        sv = torch.linalg.svdvals(w_2d)
    except RuntimeError:
        return 1.0

    # Normalize singular values to get probability distribution
    sv_sum = sv.sum().item()
    if sv_sum < eps:
        return 1.0

    p = sv / sv_sum

    # Filter out near-zero probabilities to avoid log(0)
    p_nonzero = p[p > eps]
    if len(p_nonzero) == 0:
        return 1.0

    # Compute entropy: -sum(p_i * log(p_i))
    entropy = -(p_nonzero * torch.log(p_nonzero)).sum().item()

    # Effective rank is exp(entropy)
    effective_rank = float(torch.exp(torch.tensor(entropy)).item())

    return effective_rank


class RankTrendDetector:
    """EMA-based trend detection for rank metrics.

    Tracks exponential moving average of rank values and warns when
    sustained downward trend is detected after warmup period.

    Attributes:
        alpha: EMA smoothing factor (lower = slower adaptation).
        threshold_pct: Warn if EMA drops by this fraction from initial.
        window: Steps before trend detection activates (warmup).
        ema: Current EMA value (None before first update).
        initial_ema: EMA value at end of warmup (baseline for comparison).
        step_count: Number of updates received.

    Example:
        >>> detector = RankTrendDetector(window=10, threshold_pct=0.2)
        >>> # Warmup phase
        >>> for _ in range(15):
        ...     detector.update(10.0)
        >>> # Now drop significantly
        >>> for _ in range(10):
        ...     warning = detector.update(5.0)
        ...     if warning:
        ...         print(warning)  # Will warn about rank drop
    """

    def __init__(
        self,
        alpha: float = 0.1,
        threshold_pct: float = 0.2,
        window: int = 100,
    ):
        """Initialize rank trend detector.

        Args:
            alpha: EMA smoothing factor. Lower values give slower,
                   more stable trend estimates. Default 0.1.
            threshold_pct: Fraction drop from initial that triggers warning.
                          0.2 means warn if EMA drops 20% below initial.
            window: Number of steps before trend detection activates.
                   This warmup prevents false alarms from initialization noise.
        """
        self.alpha = alpha
        self.threshold_pct = threshold_pct
        self.window = window

        self.ema: Optional[float] = None
        self.initial_ema: Optional[float] = None
        self.step_count: int = 0

    def update(self, value: float) -> Optional[str]:
        """Update detector with new rank value.

        Args:
            value: New rank measurement (stable_rank or effective_rank).

        Returns:
            Warning string if trend indicates rank collapse, else None.
            No warnings during warmup period.
        """
        self.step_count += 1

        # Update EMA
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        # Still in warmup
        if self.step_count < self.window:
            return None

        # Record initial EMA at end of warmup
        if self.step_count == self.window:
            self.initial_ema = self.ema
            return None

        # Check for sustained drop
        if self.initial_ema is not None and self.initial_ema > 0:
            drop_pct = (self.initial_ema - self.ema) / self.initial_ema
            if drop_pct > self.threshold_pct:
                return (
                    f"WARN: Rank drop detected - EMA {self.ema:.2f} is "
                    f"{drop_pct:.1%} below initial {self.initial_ema:.2f}"
                )

        return None

    def reset(self) -> None:
        """Clear all tracking state."""
        self.ema = None
        self.initial_ema = None
        self.step_count = 0


class RankHealthMonitor:
    """Per-layer rank health monitoring for neural networks.

    Computes stable rank, effective rank, and spectral norm for all
    weight matrices in a model. Tracks trends and warns on collapse.

    Classifier-Specific Monitoring:
        Critical layers like lm_head and c_proj receive stricter threshold
        monitoring. With default settings (warn_threshold=0.3, multiplier=0.5),
        classifiers warn at 15% drop vs 30% for other layers.

    Attributes:
        log_interval: Compute rank every N steps.
        warn_threshold: Fraction drop that triggers warning for non-critical layers.
        critical_layers: Layer name patterns to prioritize for warnings.
        critical_threshold_multiplier: Multiplier for critical layer thresholds.
            Critical layers use warn_threshold * critical_threshold_multiplier.

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))
        >>> monitor = RankHealthMonitor()
        >>> ranks = monitor.compute_layer_ranks(model)
        >>> for name, metrics in ranks.items():
        ...     print(f"{name}: stable={metrics['stable_rank']:.2f}")
        >>> # Check thresholds
        >>> monitor.get_threshold_for_layer("lm_head.weight")  # Returns 0.15
        >>> monitor.get_threshold_for_layer("encoder.layer.weight")  # Returns 0.3
    """

    def __init__(
        self,
        log_interval: int = 100,
        warn_threshold: float = 0.3,
        critical_layers: Optional[List[str]] = None,
        critical_threshold_multiplier: float = 0.5,
    ):
        """Initialize rank health monitor.

        Args:
            log_interval: Compute rank every N steps.
            warn_threshold: Fraction drop from initial that triggers warning.
            critical_layers: Layer name patterns to prioritize (e.g., ["lm_head"]).
                           Defaults to ["lm_head", "c_proj"].
            critical_threshold_multiplier: Multiplier applied to warn_threshold
                for critical layers. Default 0.5 means critical layers warn at
                half the threshold (e.g., 0.3 * 0.5 = 0.15 for classifiers).
        """
        self.log_interval = log_interval
        self.warn_threshold = warn_threshold
        self.critical_layers = critical_layers or ["lm_head", "c_proj"]
        self.critical_threshold_multiplier = critical_threshold_multiplier

        # Per-layer trend detectors
        self._detectors: Dict[str, RankTrendDetector] = {}

    def compute_layer_ranks(
        self, model: nn.Module
    ) -> Dict[str, Dict[str, float]]:
        """Compute rank metrics for all weight matrices.

        Args:
            model: PyTorch model to analyze.

        Returns:
            Dictionary mapping parameter names to metric dictionaries.
            Each metric dict contains:
              - stable_rank: Stable rank value
              - effective_rank: Effective rank value
              - spectral_norm: Largest singular value (||W||_2)
        """
        ranks: Dict[str, Dict[str, float]] = {}

        for name, param in model.named_parameters():
            # Skip 1D parameters (biases, LayerNorm scales)
            if param.dim() < 2:
                continue

            # Skip non-weight parameters
            weight = param.data.detach()

            # Compute metrics
            stable_rank = compute_stable_rank(weight)
            effective_rank = compute_effective_rank(weight)

            # Spectral norm is largest singular value
            try:
                w_2d = weight.view(weight.size(0), -1).float()
                sv = torch.linalg.svdvals(w_2d)
                spectral_norm = sv[0].item()
            except RuntimeError:
                spectral_norm = 0.0

            ranks[name] = {
                "stable_rank": stable_rank,
                "effective_rank": effective_rank,
                "spectral_norm": spectral_norm,
            }

        return ranks

    def get_threshold_for_layer(self, name: str) -> float:
        """Return warning threshold for a layer.

        Critical layers (lm_head, c_proj by default) use a stricter threshold
        to provide earlier warning of rank collapse in output-critical layers.

        Args:
            name: Layer parameter name.

        Returns:
            Warning threshold (fraction drop that triggers warning).
            - critical_layers: warn_threshold * critical_threshold_multiplier
            - other layers: warn_threshold

        Example:
            >>> monitor = RankHealthMonitor(warn_threshold=0.3, critical_threshold_multiplier=0.5)
            >>> monitor.get_threshold_for_layer("lm_head.weight")  # Returns 0.15
            >>> monitor.get_threshold_for_layer("encoder.layer.0.weight")  # Returns 0.3
        """
        is_critical = any(pattern in name for pattern in self.critical_layers)
        if is_critical:
            return self.warn_threshold * self.critical_threshold_multiplier
        return self.warn_threshold

    def check_warnings(
        self, ranks: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Check for rank collapse warnings.

        Updates internal trend detectors and returns warnings for
        layers showing sustained rank degradation.

        Args:
            ranks: Dictionary from compute_layer_ranks().

        Returns:
            List of warning strings (empty if no warnings).
        """
        warnings: List[str] = []

        for name, metrics in ranks.items():
            # Get or create detector for this layer
            if name not in self._detectors:
                threshold = self.get_threshold_for_layer(name)
                self._detectors[name] = RankTrendDetector(
                    threshold_pct=threshold, window=self.log_interval
                )

            detector = self._detectors[name]

            # Update with stable rank (more robust than effective rank)
            warning = detector.update(metrics["stable_rank"])
            if warning:
                warnings.append(f"[{name}] {warning}")

        return warnings


__all__ = [
    "compute_stable_rank",
    "compute_effective_rank",
    "RankTrendDetector",
    "RankHealthMonitor",
]
