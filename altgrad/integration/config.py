"""Per-layer precision configuration for mixed-precision quantization.

Provides LayerPrecisionRule and QuantizationConfig for specifying
which layers use which FP8 format (or remain in BF16).

This enables experiments like:
- Attention in BF16 (full precision) + MLP in FP8
- Different FP8 formats for different layer types
- Gradual precision reduction (higher layers more quantized)

Example:
    >>> from altgrad.integration import QuantizationConfig, LayerPrecisionRule
    >>> config = QuantizationConfig(
    ...     default_format="E5M2",
    ...     layer_rules=[
    ...         LayerPrecisionRule(r"\\.attn\\.", None),     # Attention BF16
    ...         LayerPrecisionRule(r"\\.mlp\\.", "E5M2"),    # MLP FP8
    ...         LayerPrecisionRule(r"lm_head", None),        # LM head BF16
    ...     ],
    ... )
    >>> format_for_layer = config.get_format_for_layer("transformer.h.0.mlp.c_fc")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from altgrad.quantization import FP8Format, FORMAT_REGISTRY


@dataclass
class LayerPrecisionRule:
    """Rule for matching layer names to FP8 formats.

    Uses regex pattern matching to identify layers by their dotted names
    (e.g., "transformer.h.0.attn.c_attn").

    Attributes:
        pattern: Regex pattern for matching layer names
        format: FP8 format name (e.g., "E5M2") or None for BF16

    Example:
        >>> rule = LayerPrecisionRule(r"\\.attn\\.", None)  # Attention in BF16
        >>> rule.matches("transformer.h.0.attn.c_attn")  # True
        >>> rule.matches("transformer.h.0.mlp.c_fc")  # False
    """

    pattern: str
    format: Optional[str]

    def matches(self, layer_name: str) -> bool:
        """Check if layer name matches this rule's pattern.

        Args:
            layer_name: Full dotted layer name (e.g., "transformer.h.0.attn.c_attn")

        Returns:
            True if pattern matches anywhere in layer_name
        """
        return re.search(self.pattern, layer_name) is not None

    def get_format(self) -> Optional[FP8Format]:
        """Get the FP8Format for this rule, or None for BF16.

        Returns:
            FP8Format instance if format is specified, None for BF16 precision
        """
        if self.format is None:
            return None
        return FORMAT_REGISTRY[self.format]


@dataclass
class QuantizationConfig:
    """Configuration for model quantization with per-layer precision.

    Supports mixed-precision training by allowing different FP8 formats
    (or full precision BF16) for different layers based on regex patterns.
    First matching rule wins.

    Attributes:
        default_format: Default FP8 format for unmatched layers (None = BF16)
        layer_rules: Ordered list of rules (first match wins)
        history_len: Amax history length for dynamic scaling
        quantize_input: Whether to quantize input activations

    Example:
        >>> config = QuantizationConfig(
        ...     default_format="E5M2",
        ...     layer_rules=[
        ...         LayerPrecisionRule(r"\\.attn\\.", None),     # Attention BF16
        ...         LayerPrecisionRule(r"\\.mlp\\.", "E5M2"),    # MLP FP8
        ...         LayerPrecisionRule(r"lm_head", None),        # LM head BF16
        ...     ],
        ... )
        >>> config.get_format_for_layer("transformer.h.0.attn.c_attn")
        None  # BF16
        >>> config.get_format_for_layer("transformer.h.0.mlp.c_fc").name
        'E5M2'
    """

    default_format: Optional[str] = "E5M2"
    layer_rules: List[LayerPrecisionRule] = field(default_factory=list)
    history_len: int = 16
    quantize_input: bool = False

    def get_format_for_layer(self, layer_name: str) -> Optional[FP8Format]:
        """Get FP8 format for a layer based on rules.

        First matching rule wins. If no rule matches, uses default_format.
        Returns None if the layer should stay in BF16 precision.

        Args:
            layer_name: Full layer name (e.g., "transformer.h.0.attn.c_attn")

        Returns:
            FP8Format for quantization, or None for BF16 precision
        """
        for rule in self.layer_rules:
            if rule.matches(layer_name):
                return rule.get_format()

        # Use default format
        if self.default_format is None:
            return None
        return FORMAT_REGISTRY[self.default_format]


def create_mixed_precision_config(
    attention_format: Optional[str] = None,
    mlp_format: str = "E5M2",
    lm_head_format: Optional[str] = None,
    default_format: str = "E5M2",
    history_len: int = 16,
    quantize_input: bool = False,
) -> QuantizationConfig:
    """Create a standard mixed-precision config for GPT models.

    Provides a convenient helper for common GPT quantization patterns:
    - Attention layers can stay in BF16 for precision-sensitive operations
    - MLP layers use FP8 (most of the compute)
    - LM head stays in BF16 (tied with embedding, affects output quality)

    Args:
        attention_format: Format for attention layers (None = BF16)
        mlp_format: Format for MLP layers (default "E5M2")
        lm_head_format: Format for LM head (None = BF16, recommended)
        default_format: Default format for unmatched layers
        history_len: Amax history length for dynamic scaling
        quantize_input: Whether to quantize input activations

    Returns:
        QuantizationConfig with typical GPT layer rules

    Example:
        >>> config = create_mixed_precision_config(
        ...     attention_format=None,  # BF16
        ...     mlp_format="E5M2",
        ...     lm_head_format=None,  # BF16
        ... )
    """
    rules = [
        # Attention patterns (c_attn for QKV, c_proj for output)
        LayerPrecisionRule(r"\.attn\.", attention_format),
        # MLP patterns
        LayerPrecisionRule(r"\.mlp\.", mlp_format),
        # LM head (usually tied with embedding)
        LayerPrecisionRule(r"lm_head", lm_head_format),
    ]

    return QuantizationConfig(
        default_format=default_format,
        layer_rules=rules,
        history_len=history_len,
        quantize_input=quantize_input,
    )


__all__ = [
    "LayerPrecisionRule",
    "QuantizationConfig",
    "create_mixed_precision_config",
]
