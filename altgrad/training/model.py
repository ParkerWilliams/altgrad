"""nanoGPT-style transformer model for language modeling.

Provides a simplified GPT architecture suitable for quantization experiments.
Forked from nanoGPT essential structure with focus on training.

Key features:
  - Causal self-attention with optional Flash Attention (PyTorch 2.0+)
  - Pre-norm transformer blocks (LayerNorm before attention/MLP)
  - Weight tying between token embedding and LM head
  - Separate weight decay for matmul weights vs biases/norms

Example:
    >>> from altgrad.training.model import GPT, GPTConfig
    >>> config = GPTConfig(n_layer=6, n_head=6, n_embd=384, vocab_size=50257)
    >>> model = GPT(config)
    >>> x = torch.randint(0, 50257, (4, 256))
    >>> logits, loss = model(x, targets=x)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GPTConfig:
    """GPT model configuration.

    Attributes:
        n_layer: Number of transformer blocks
        n_head: Number of attention heads
        n_embd: Embedding/hidden dimension
        block_size: Maximum sequence length (context window)
        vocab_size: Vocabulary size
        dropout: Dropout probability (default 0.0)
        bias: Use bias in linear layers and LayerNorm (default True)
    """

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    vocab_size: int = 50304
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Implements scaled dot-product attention with causal masking.
    Uses torch.nn.functional.scaled_dot_product_attention when available
    (PyTorch 2.0+) for fused attention, otherwise falls back to manual.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads (combined)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Check for Flash Attention support
        self.flash = hasattr(F, "scaled_dot_product_attention")

        if not self.flash:
            # Manual attention needs causal mask
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()  # batch, seq_len, embedding dim

        # Compute q, k, v for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, nh, T, hs)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if self.flash:
            # Efficient fused attention (PyTorch 2.0+)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention with causal mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble all head outputs: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Two-layer feedforward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-norm architecture.

    LayerNorm -> Attention -> residual
    LayerNorm -> MLP -> residual
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT language model.

    Token + position embeddings -> Transformer blocks -> LayerNorm -> LM head
    Weight tying: LM head shares weights with token embedding.

    Example:
        >>> config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        >>> model = GPT(config)
        >>> x = torch.randint(0, 100, (2, 32))
        >>> logits, loss = model(x, targets=x)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        # LM head (weight tied to token embedding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass for language modeling.

        Args:
            idx: Input token indices, shape (batch, seq_len)
            targets: Target token indices for loss computation (same shape)

        Returns:
            Tuple of (logits, loss):
              - logits: shape (batch, seq_len, vocab_size)
              - loss: scalar cross-entropy loss, or None if no targets
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} exceeds block_size {self.config.block_size}"
        )

        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # LM head
        logits = self.lm_head(x)  # (b, t, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Configure AdamW optimizer with weight decay separation.

        Weight decay is applied only to 2D parameters (matmul weights).
        1D parameters (biases, LayerNorm) are not decayed.

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta coefficients (beta1, beta2)
            device_type: Device type for fused optimizer ("cuda" enables fused)

        Returns:
            Configured AdamW optimizer
        """
        # Separate parameters into decay and no-decay groups
        decay = set()
        no_decay = set()

        for pn, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # 2D params (weights) get decay, 1D params (biases, norms) don't
            if p.dim() >= 2:
                decay.add(pn)
            else:
                no_decay.add(pn)

        # Validate all parameters accounted for
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Params in both sets: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, "Missing params"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        # Use fused AdamW on CUDA if available
        use_fused = device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused,
        )

        return optimizer


__all__ = [
    "GPTConfig",
    "GPT",
    "CausalSelfAttention",
    "MLP",
    "Block",
]
