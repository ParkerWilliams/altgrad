"""Manifold-aware optimizer with stiffness-preconditioned updates.

Implements ManifoldAdamW: AdamW with optional stiffness preconditioning
that scales gradients by local FP8 grid spacing, causing updates to move
weights by consistent ULP counts rather than fixed real values.

Requirements covered:
  - MANI-02: Stiffness-preconditioned gradient step
  - MANI-03: Standard vs manifold-aware training mode toggle
  - MANI-04: Bit-position tracking (latent integer state)

Example:
    >>> from altgrad.training.optimizer import ManifoldAdamW
    >>> optimizer = ManifoldAdamW(model.parameters(), lr=3e-4, manifold_aware=True)
    >>> optimizer.step()
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer

from altgrad.quantization.advanced_diagnostics import compute_stiffness_field


class ManifoldAdamW(Optimizer):
    """AdamW optimizer with optional stiffness-preconditioned updates.

    When manifold_aware=True, multiplies gradients by the local stiffness
    factor before computing Adam moments. This causes updates to move
    weights by a consistent number of ULPs rather than fixed real values.

    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (default: 1e-3)
        betas: Adam beta coefficients (default: (0.9, 0.999))
        eps: Numerical stability epsilon (default: 1e-8)
        weight_decay: Decoupled weight decay (default: 0.01)
        manifold_aware: Enable stiffness preconditioning (default: True)
        mantissa_bits: Format mantissa bits M for S = 2^(floor(log2|w|) - M) (default: 2 for E5M2)
        max_stiffness: Maximum stiffness clamp to prevent explosion (default: 1e6)

    Example:
        >>> optimizer = ManifoldAdamW(
        ...     model.parameters(),
        ...     lr=3e-4,
        ...     manifold_aware=True,
        ...     mantissa_bits=2,  # E5M2
        ... )
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        manifold_aware: bool = True,
        mantissa_bits: int = 2,
        max_stiffness: float = 1e6,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            manifold_aware=manifold_aware,
            mantissa_bits=mantissa_bits,
            max_stiffness=max_stiffness,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns loss (optional)

        Returns:
            Loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ManifoldAdamW does not support sparse gradients")

                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Bit-position tracking (MANI-04)
                    state["bit_position"] = torch.zeros_like(p)

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Stiffness preconditioning (MANI-02)
                if group["manifold_aware"]:
                    stiffness = compute_stiffness_field(p.data, group["mantissa_bits"])
                    # Handle NaN (zero weights have undefined stiffness)
                    stiffness = torch.where(
                        torch.isnan(stiffness),
                        torch.ones_like(stiffness),
                        stiffness
                    )
                    # Clamp to prevent explosion at large magnitudes
                    stiffness = stiffness.clamp(max=group["max_stiffness"])
                    # Precondition gradient
                    grad = grad * stiffness

                # Store weight before update (for bit-position tracking)
                weight_before = p.data.clone()

                # Decoupled weight decay (AdamW style)
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Update bit-position tracking (MANI-04)
                self._update_bit_position(state, weight_before, p.data)

        return loss

    def _update_bit_position(
        self,
        state: dict,
        before: Tensor,
        after: Tensor,
    ) -> None:
        """Track cumulative ULP movement for MANI-04.

        Args:
            state: Per-parameter optimizer state
            before: Weight values before update
            after: Weight values after update
        """
        # ULP at each position = distance to next representable value
        inf_tensor = torch.full_like(before, float("inf"))
        ulp = torch.abs(torch.nextafter(before, inf_tensor) - before)

        # Avoid division by zero for very small values
        safe_ulp = ulp.clamp(min=1e-45)

        # Signed ULP movement (positive = increased, negative = decreased)
        delta_ulps = (after - before) / safe_ulp

        # Accumulate
        state["bit_position"] += delta_ulps


class GridOptim:
    """Grid-based optimizer with FP32 master weights and stochastic rounding.

    Maintains FP32 master weights and projects to FP8 grid using stochastic
    rounding in rung space. This approach gives explicit control over discrete
    weight transitions.

    Args:
        params: Iterable of model parameters
        scale: Learning rate in rung units (default: 6.0)
        momentum: SGD momentum coefficient (default: 0.9)
        weight_decay: L2 penalty coefficient (default: 1e-4)
        rung_clip: Maximum rung movement per step (default: 10)
        fp8_dtype: FP8 dtype for grid construction (default: torch.float8_e4m3fn)
        device: Device for grid tensor (default: "cuda" if available)

    Example:
        >>> optimizer = GridOptim(model.parameters(), scale=6.0)
        >>> flips, updates = optimizer.step()
    """

    def __init__(
        self,
        params,
        scale: float = 6.0,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        rung_clip: int = 10,
        fp8_dtype=None,
        device=None,
    ):
        self.params = list(params)
        self.scale = scale
        self.momentum = momentum
        self.wd = weight_decay
        self.rung_clip = rung_clip

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Create FP32 master weights (copy of parameters)
        self.master_p = [p.detach().clone().float().to(device) for p in self.params]

        # Initialize velocity (momentum buffer)
        self.velocity = [torch.zeros_like(p) for p in self.master_p]

        # Build FP8 grid from all representable values
        fp8_dtype = fp8_dtype or getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is not None:
            raw_bits = torch.arange(-128, 128, dtype=torch.int8)
            all_floats = raw_bits.view(fp8_dtype).to(torch.float32)
            clean = all_floats[~torch.isnan(all_floats) & ~torch.isinf(all_floats)]
            self.grid = torch.sort(torch.unique(clean))[0].to(device)
        else:
            self.grid = None  # Fallback for testing without FP8 support

    @torch.no_grad()
    def step(self, current_scale: float | None = None) -> tuple[int, int]:
        """Perform single optimization step.

        Args:
            current_scale: Override scale for this step (e.g., for schedules)

        Returns:
            Tuple of (flips, updates) where:
            - flips: Number of FP8 values that changed
            - updates: Number of non-zero gradient elements applied
        """
        flips, updates = 0, 0
        scale = current_scale if current_scale is not None else self.scale

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            old_data = p.data.clone()
            grad = p.grad.to(torch.float32)

            # Count updates (non-zero gradient elements)
            updates += (grad.abs() > 1e-10).sum().item()

            # Weight decay
            if self.wd != 0:
                grad.add_(self.master_p[i], alpha=self.wd)

            # Momentum
            self.velocity[i] = self.momentum * self.velocity[i] + grad

            if self.grid is not None:
                # Grid-based update
                indices = torch.searchsorted(self.grid, self.master_p[i].contiguous())
                v_rungs = self.velocity[i] * scale

                # CRITICAL: Clip to prevent NaN at grid boundaries
                v_rungs = torch.clamp(v_rungs, -self.rung_clip, self.rung_clip)

                # Stochastic rounding
                v_rounded = torch.floor(v_rungs + torch.rand_like(v_rungs)).to(torch.int32)

                # New grid indices (subtract for descent)
                new_indices = torch.clamp(indices - v_rounded, 0, len(self.grid) - 1)
                new_floats = self.grid[new_indices.long()].view(p.shape)

                self.master_p[i].copy_(new_floats)
                p.data.copy_(new_floats)
            else:
                # Fallback: Euclidean update
                self.master_p[i].sub_(scale * self.velocity[i])
                p.data.copy_(self.master_p[i])

            flips += (p.data != old_data).sum().item()

        return flips, updates

    def zero_grad(self) -> None:
        """Zero all parameter gradients."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


__all__ = [
    "ManifoldAdamW",
    "GridOptim",
]
