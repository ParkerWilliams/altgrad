"""Classification model with FP8 quantization support.

Transformer encoder with classification head for multi-label classification.
Supports simulated FP8 quantization on encoder layers while keeping
classification head in full precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoConfig


@dataclass
class ClassifierConfig:
    """Configuration for classification model."""

    # Model architecture
    encoder_name: str = "distilbert-base-uncased"
    num_labels: int = 100
    hidden_dropout: float = 0.1
    classifier_dropout: float = 0.1

    # Quantization
    quantize_encoder: bool = True
    quantize_classifier: bool = False  # Keep classifier in full precision

    # Loss function
    pos_weight: float = 1.0  # Weight for positive class (higher = penalize FN more)


class TransformerClassifier(nn.Module):
    """Transformer encoder with multi-label classification head.

    Uses a pretrained transformer encoder (DistilBERT, RoBERTa, etc.)
    with a linear classification head for multi-label prediction.

    The encoder can be quantized to FP8 while keeping the classification
    head in full precision for stable output predictions.

    Attributes:
        config: Classifier configuration
        encoder: Pretrained transformer encoder
        classifier: Linear classification head
        dropout: Dropout before classifier

    Example:
        >>> config = ClassifierConfig(num_labels=1000)
        >>> model = TransformerClassifier(config)
        >>> logits = model(input_ids, attention_mask)
        >>> loss = model.compute_loss(logits, labels)
    """

    def __init__(self, config: ClassifierConfig):
        """Initialize classifier.

        Args:
            config: Classifier configuration
        """
        super().__init__()
        self.config = config

        # Load pretrained encoder
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden_size = self.encoder.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

        # Loss function for multi-label with asymmetric weighting
        # pos_weight > 1 penalizes false negatives more heavily
        if config.pos_weight != 1.0:
            pos_weight = torch.full((config.num_labels,), config.pos_weight)
            self.register_buffer("pos_weight", pos_weight)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Track which parameters to quantize
        self._quantize_param_names = set()
        if config.quantize_encoder:
            for name, _ in self.encoder.named_parameters():
                self._quantize_param_names.add(f"encoder.{name}")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Multi-hot labels (batch, num_labels), optional

        Returns:
            Tuple of (logits, loss) where loss is None if labels not provided
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Pool: use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return logits, loss

    def compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute BCE loss for multi-label classification.

        Args:
            logits: Raw model outputs (batch, num_labels)
            labels: Multi-hot labels (batch, num_labels)

        Returns:
            Scalar loss tensor
        """
        return self.loss_fn(logits, labels)

    def should_quantize(self, param_name: str) -> bool:
        """Check if parameter should be quantized.

        Args:
            param_name: Full parameter name

        Returns:
            True if parameter should be FP8 quantized
        """
        return param_name in self._quantize_param_names

    def get_quantizable_params(self) -> Dict[str, nn.Parameter]:
        """Get parameters that should be quantized.

        Returns:
            Dictionary mapping parameter names to parameters
        """
        params = {}
        for name, param in self.named_parameters():
            if self.should_quantize(name) and param.dim() >= 2:
                params[name] = param
        return params

    def configure_optimizers(
        self,
        weight_decay: float = 0.01,
        learning_rate: float = 2e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
    ) -> torch.optim.AdamW:
        """Configure AdamW optimizer with weight decay separation.

        2D parameters (weights) get weight decay, 1D parameters (biases,
        layer norms) don't.

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters

        Returns:
            Configured AdamW optimizer
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


__all__ = [
    "ClassifierConfig",
    "TransformerClassifier",
]
