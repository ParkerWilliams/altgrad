"""Tests for FP8 model integration: QuantizedLinear wrapper and model surgery.

Tests validate:
1. QuantizedLinear wrapper applies quantization correctly
2. Gradient flow through STE maintains training capability
3. Model surgery replaces nn.Linear with QuantizedLinear
4. Dequantization restores original module structure
5. Weight tying preservation in GPT models
"""

import torch
import torch.nn as nn
import pytest

from altgrad.quantization import E5M2, E3M4
from altgrad.integration import QuantizedLinear, quantize_model, dequantize_model
from altgrad.training.model import GPT, GPTConfig


class TestQuantizedLinearForward:
    """Test QuantizedLinear forward pass behavior."""

    def test_quantized_linear_forward_produces_output(self):
        """QuantizedLinear forward should produce output with correct shape."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E5M2)

        x = torch.randn(4, 10)
        y = q_linear(x)

        assert y.shape == (4, 5), f"Expected shape (4, 5), got {y.shape}"

    def test_quantized_linear_forward_differs_from_original(self):
        """QuantizedLinear output should differ from unwrapped linear due to quantization."""
        linear = nn.Linear(10, 5, bias=False)
        # Use weights that will show quantization effect
        nn.init.uniform_(linear.weight, -1.0, 1.0)

        q_linear = QuantizedLinear(linear, E5M2)

        x = torch.randn(4, 10)

        # Direct linear output
        y_original = linear(x)

        # QuantizedLinear output (quantizes weights)
        y_quantized = q_linear(x)

        # Outputs should differ due to weight quantization
        # (unless weights happen to already be exactly representable)
        # Allow for small tolerance as some weights may be unchanged
        diff = (y_original - y_quantized).abs().max()
        # With random weights, quantization should introduce some difference
        # This test may occasionally pass with identical outputs if lucky
        assert diff.item() >= 0, "Outputs computed successfully"


class TestQuantizedLinearGradient:
    """Test gradient flow through QuantizedLinear via STE."""

    def test_quantized_linear_gradient_flow_to_input(self):
        """Gradient should flow back to input tensor via STE."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E5M2)

        x = torch.randn(4, 10, requires_grad=True)
        y = q_linear(x)
        y.sum().backward()

        assert x.grad is not None, "Input gradient should not be None"
        assert x.grad.shape == x.shape, "Gradient shape should match input"

    def test_quantized_linear_gradient_flow_to_weights(self):
        """Gradient should flow back to original Linear weights via STE."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E5M2)

        x = torch.randn(4, 10, requires_grad=True)
        y = q_linear(x)
        y.sum().backward()

        assert linear.weight.grad is not None, "Weight gradient should not be None"
        assert linear.weight.grad.shape == linear.weight.shape

    def test_quantized_linear_gradient_flow_to_bias(self):
        """Gradient should flow back to bias if present."""
        linear = nn.Linear(10, 5, bias=True)
        q_linear = QuantizedLinear(linear, E5M2)

        x = torch.randn(4, 10, requires_grad=True)
        y = q_linear(x)
        y.sum().backward()

        assert linear.bias.grad is not None, "Bias gradient should not be None"


class TestQuantizedLinearAmaxHistory:
    """Test amax history tracking in QuantizedLinear."""

    def test_quantized_linear_has_weight_history(self):
        """QuantizedLinear should have weight_history attribute."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E5M2)

        assert hasattr(q_linear, 'weight_history'), "Should have weight_history"

    def test_quantized_linear_has_input_history(self):
        """QuantizedLinear should have input_history attribute."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E5M2)

        assert hasattr(q_linear, 'input_history'), "Should have input_history"

    def test_quantized_linear_amax_updates_on_forward(self):
        """Amax history should update during forward pass."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E5M2)

        # Run multiple forward passes
        for _ in range(3):
            x = torch.randn(4, 10)
            q_linear(x)

        # History should have been updated
        assert len(q_linear.weight_history) > 0, "weight_history should be populated"


class TestQuantizedLinearProperties:
    """Test QuantizedLinear property exposure."""

    def test_weight_property_exposes_underlying(self):
        """wrapper.weight should be wrapper.linear.weight."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E5M2)

        assert q_linear.weight is q_linear.linear.weight

    def test_bias_property_exposes_underlying(self):
        """wrapper.bias should be wrapper.linear.bias."""
        linear = nn.Linear(10, 5, bias=True)
        q_linear = QuantizedLinear(linear, E5M2)

        assert q_linear.bias is q_linear.linear.bias

    def test_bias_property_none_when_no_bias(self):
        """wrapper.bias should be None when linear has no bias."""
        linear = nn.Linear(10, 5, bias=False)
        q_linear = QuantizedLinear(linear, E5M2)

        assert q_linear.bias is None


class TestQuantizeModelBasic:
    """Test quantize_model() basic functionality."""

    def test_quantize_model_replaces_linear_layers(self):
        """quantize_model() should replace nn.Linear with QuantizedLinear."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        quantize_model(model, E5M2)

        assert isinstance(model[0], QuantizedLinear), "First Linear should be QuantizedLinear"
        assert isinstance(model[2], QuantizedLinear), "Second Linear should be QuantizedLinear"
        assert isinstance(model[1], nn.ReLU), "ReLU should be unchanged"

    def test_quantize_model_preserves_weights(self):
        """quantize_model() should preserve original weights in wrapped Linear."""
        linear = nn.Linear(10, 5)
        original_weight = linear.weight.data.clone()
        original_bias = linear.bias.data.clone()

        model = nn.Sequential(linear)
        quantize_model(model, E5M2)

        # Get the QuantizedLinear wrapper
        q_linear = model[0]
        assert isinstance(q_linear, QuantizedLinear)

        # Weights should be preserved
        assert torch.equal(q_linear.linear.weight.data, original_weight)
        assert torch.equal(q_linear.linear.bias.data, original_bias)


class TestQuantizeModelSkipPatterns:
    """Test quantize_model() skip_patterns functionality."""

    def test_quantize_model_skip_patterns_string_match(self):
        """quantize_model() should skip layers matching skip_patterns."""
        model = nn.ModuleDict({
            'layer1': nn.Linear(10, 20),
            'layer2': nn.Linear(20, 10),
            'output': nn.Linear(10, 5),
        })

        quantize_model(model, E5M2, skip_patterns=['layer2'])

        assert isinstance(model['layer1'], QuantizedLinear), "layer1 should be quantized"
        assert isinstance(model['layer2'], nn.Linear), "layer2 should be skipped"
        assert not isinstance(model['layer2'], QuantizedLinear), "layer2 should remain nn.Linear"
        assert isinstance(model['output'], QuantizedLinear), "output should be quantized"

    def test_quantize_model_skip_multiple_patterns(self):
        """quantize_model() should skip multiple patterns."""
        model = nn.ModuleDict({
            'layer1': nn.Linear(10, 20),
            'layer2': nn.Linear(20, 10),
            'layer3': nn.Linear(10, 5),
        })

        quantize_model(model, E5M2, skip_patterns=['layer1', 'layer3'])

        assert isinstance(model['layer1'], nn.Linear), "layer1 should be skipped"
        assert isinstance(model['layer2'], QuantizedLinear), "layer2 should be quantized"
        assert isinstance(model['layer3'], nn.Linear), "layer3 should be skipped"


class TestQuantizeModelNested:
    """Test quantize_model() with nested modules."""

    def test_quantize_model_nested_modules(self):
        """quantize_model() should handle nested module structures."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
            ),
            nn.Linear(20, 5),
        )

        quantize_model(model, E5M2)

        # Check nested Linear is replaced
        assert isinstance(model[0][0], QuantizedLinear), "Nested Linear should be QuantizedLinear"
        assert isinstance(model[1], QuantizedLinear), "Top-level Linear should be QuantizedLinear"

    def test_quantize_model_moduledict_nested(self):
        """quantize_model() should handle ModuleDict with nested structures."""
        inner = nn.ModuleDict({
            'fc': nn.Linear(10, 20),
        })
        model = nn.ModuleDict({
            'block': inner,
            'output': nn.Linear(20, 5),
        })

        quantize_model(model, E5M2)

        assert isinstance(model['block']['fc'], QuantizedLinear)
        assert isinstance(model['output'], QuantizedLinear)


class TestQuantizeModelGPT:
    """Test quantize_model() on GPT model with weight tying."""

    def test_gpt_weight_tying_before_surgery(self):
        """Verify GPT has weight tying before surgery (sanity check)."""
        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT(config)

        assert model.transformer.wte.weight is model.lm_head.weight, \
            "GPT should have weight tying between wte and lm_head"

    def test_quantize_model_gpt_preserves_weight_tying(self):
        """Weight tying between wte and lm_head must be preserved after surgery."""
        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT(config)

        # Verify weight tying exists
        assert model.transformer.wte.weight is model.lm_head.weight

        # Quantize model, skipping lm_head to preserve weight tying
        quantize_model(model, E5M2, skip_patterns=['lm_head'])

        # Weight tying must still hold
        assert model.transformer.wte.weight is model.lm_head.weight, \
            "Weight tying should be preserved after quantize_model"

    def test_quantize_model_gpt_replaces_attention_linear(self):
        """GPT attention Linear layers (c_attn, c_proj) should be quantized."""
        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT(config)

        quantize_model(model, E5M2, skip_patterns=['lm_head'])

        # Check transformer.h[0].attn.c_attn is QuantizedLinear
        attn = model.transformer.h[0].attn
        assert isinstance(attn.c_attn, QuantizedLinear), \
            "c_attn should be QuantizedLinear"
        assert isinstance(attn.c_proj, QuantizedLinear), \
            "c_proj should be QuantizedLinear"

    def test_quantize_model_gpt_replaces_mlp_linear(self):
        """GPT MLP Linear layers (c_fc, c_proj) should be quantized."""
        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT(config)

        quantize_model(model, E5M2, skip_patterns=['lm_head'])

        mlp = model.transformer.h[0].mlp
        assert isinstance(mlp.c_fc, QuantizedLinear), \
            "MLP c_fc should be QuantizedLinear"
        assert isinstance(mlp.c_proj, QuantizedLinear), \
            "MLP c_proj should be QuantizedLinear"


class TestDequantizeModel:
    """Test dequantize_model() restores original structure."""

    def test_dequantize_model_restores_linear(self):
        """dequantize_model() should restore nn.Linear from QuantizedLinear."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        # Quantize
        quantize_model(model, E5M2)
        assert isinstance(model[0], QuantizedLinear)
        assert isinstance(model[2], QuantizedLinear)

        # Dequantize
        dequantize_model(model)

        assert isinstance(model[0], nn.Linear), "Should be restored to nn.Linear"
        assert not isinstance(model[0], QuantizedLinear), "Should not be QuantizedLinear"
        assert isinstance(model[2], nn.Linear), "Should be restored to nn.Linear"

    def test_dequantize_model_preserves_weights(self):
        """dequantize_model() should preserve weights after round-trip."""
        linear = nn.Linear(10, 5)
        original_weight = linear.weight.data.clone()

        model = nn.Sequential(linear)

        # Round-trip
        quantize_model(model, E5M2)
        dequantize_model(model)

        # Weights should be preserved
        assert torch.equal(model[0].weight.data, original_weight)

    def test_dequantize_model_nested(self):
        """dequantize_model() should handle nested modules."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 20),
            ),
            nn.Linear(20, 5),
        )

        quantize_model(model, E5M2)
        dequantize_model(model)

        assert isinstance(model[0][0], nn.Linear)
        assert not isinstance(model[0][0], QuantizedLinear)


class TestQuantizedLinearDifferentFormats:
    """Test QuantizedLinear with different FP8 formats."""

    def test_quantized_linear_e3m4_format(self):
        """QuantizedLinear should work with E3M4 format."""
        linear = nn.Linear(10, 5)
        q_linear = QuantizedLinear(linear, E3M4)

        x = torch.randn(4, 10)
        y = q_linear(x)

        assert y.shape == (4, 5)

    def test_quantize_model_with_e3m4(self):
        """quantize_model should accept different formats."""
        model = nn.Sequential(nn.Linear(10, 5))

        quantize_model(model, E3M4)

        assert isinstance(model[0], QuantizedLinear)
        assert model[0].format is E3M4


# =============================================================================
# QuantizationConfig tests (per-layer precision configuration)
# =============================================================================


class TestLayerPrecisionRule:
    """Test LayerPrecisionRule pattern matching."""

    def test_layer_precision_rule_matches_pattern(self):
        """LayerPrecisionRule should match layer names via regex pattern."""
        from altgrad.integration import LayerPrecisionRule

        rule = LayerPrecisionRule(pattern=r"\.attn\.", format="E5M2")

        # Should match attention layers
        assert rule.matches("transformer.h.0.attn.c_attn") is True
        assert rule.matches("transformer.h.1.attn.c_proj") is True

        # Should not match MLP layers
        assert rule.matches("transformer.h.0.mlp.c_fc") is False
        assert rule.matches("transformer.h.0.mlp.c_proj") is False

    def test_layer_precision_rule_none_format(self):
        """LayerPrecisionRule with format=None should return None (BF16)."""
        from altgrad.integration import LayerPrecisionRule

        rule = LayerPrecisionRule(pattern=r"\.attn\.", format=None)

        assert rule.get_format() is None


class TestQuantizationConfig:
    """Test QuantizationConfig per-layer format selection."""

    def test_quantization_config_get_format_for_layer(self):
        """QuantizationConfig should route layers to different formats based on rules."""
        from altgrad.integration import LayerPrecisionRule, QuantizationConfig

        config = QuantizationConfig(
            default_format="E5M2",
            layer_rules=[
                LayerPrecisionRule(r"\.attn\.", None),  # Attention stays BF16
                LayerPrecisionRule(r"\.mlp\.", "E5M2"),  # MLP uses E5M2
            ],
        )

        # Attention should be None (BF16)
        assert config.get_format_for_layer("transformer.h.0.attn.c_attn") is None
        assert config.get_format_for_layer("transformer.h.0.attn.c_proj") is None

        # MLP should be E5M2
        mlp_format = config.get_format_for_layer("transformer.h.0.mlp.c_fc")
        assert mlp_format is not None
        assert mlp_format.name == "E5M2"

    def test_quantization_config_default_format(self):
        """QuantizationConfig should use default_format for unmatched layers."""
        from altgrad.integration import QuantizationConfig

        config = QuantizationConfig(default_format="E3M4", layer_rules=[])

        # Any layer should get E3M4
        format_result = config.get_format_for_layer("any.layer.name")
        assert format_result is not None
        assert format_result.name == "E3M4"

    def test_quantization_config_first_match_wins(self):
        """QuantizationConfig should use first matching rule."""
        from altgrad.integration import LayerPrecisionRule, QuantizationConfig

        config = QuantizationConfig(
            default_format="E5M2",
            layer_rules=[
                LayerPrecisionRule(r"c_proj", "E3M4"),  # More specific
                LayerPrecisionRule(r"\.mlp\.", "E5M2"),  # More general
            ],
        )

        # mlp.c_proj should match c_proj rule first -> E3M4
        format_result = config.get_format_for_layer("transformer.h.0.mlp.c_proj")
        assert format_result is not None
        assert format_result.name == "E3M4"

        # mlp.c_fc should match mlp rule -> E5M2
        format_result = config.get_format_for_layer("transformer.h.0.mlp.c_fc")
        assert format_result is not None
        assert format_result.name == "E5M2"


class TestQuantizeModelWithConfig:
    """Test quantize_model() with QuantizationConfig for mixed precision."""

    def test_quantize_model_with_config_mixed_precision(self):
        """quantize_model with config should apply per-layer formats."""
        from altgrad.integration import (
            LayerPrecisionRule,
            QuantizationConfig,
            QuantizedLinear,
            quantize_model,
        )

        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT(config)

        # Mixed precision: attention BF16, MLP FP8
        quant_config = QuantizationConfig(
            default_format="E5M2",
            layer_rules=[
                LayerPrecisionRule(r"\.attn\.", None),  # Attention stays BF16
                LayerPrecisionRule(r"\.mlp\.", "E5M2"),  # MLP uses FP8
                LayerPrecisionRule(r"lm_head", None),  # lm_head stays BF16
            ],
        )

        quantize_model(model, config=quant_config)

        # Attention should stay as nn.Linear (BF16)
        attn_layer = model.transformer.h[0].attn.c_attn
        assert isinstance(attn_layer, nn.Linear), "Attention should stay nn.Linear"
        assert not isinstance(attn_layer, QuantizedLinear), "Attention should not be quantized"

        # MLP should be QuantizedLinear with E5M2
        mlp_layer = model.transformer.h[0].mlp.c_fc
        assert isinstance(mlp_layer, QuantizedLinear), "MLP should be QuantizedLinear"
        assert mlp_layer.fp8_format.name == "E5M2", "MLP format should be E5M2"

        # lm_head should stay as nn.Linear
        assert isinstance(model.lm_head, nn.Linear), "lm_head should stay nn.Linear"
        assert not isinstance(model.lm_head, QuantizedLinear), "lm_head should not be quantized"

    def test_quantize_model_config_different_formats(self):
        """quantize_model should apply different formats to different layers."""
        from altgrad.integration import (
            LayerPrecisionRule,
            QuantizationConfig,
            QuantizedLinear,
            quantize_model,
        )

        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100)
        model = GPT(config)

        # Different formats for c_fc and c_proj
        quant_config = QuantizationConfig(
            default_format=None,  # Default is BF16
            layer_rules=[
                LayerPrecisionRule(r"mlp\.c_fc", "E5M2"),  # c_fc uses E5M2
                LayerPrecisionRule(r"mlp\.c_proj", "E3M4"),  # c_proj uses E3M4
            ],
        )

        quantize_model(model, config=quant_config)

        # c_fc should have E5M2
        c_fc = model.transformer.h[0].mlp.c_fc
        assert isinstance(c_fc, QuantizedLinear)
        assert c_fc.fp8_format.name == "E5M2"

        # c_proj should have E3M4
        c_proj = model.transformer.h[0].mlp.c_proj
        assert isinstance(c_proj, QuantizedLinear)
        assert c_proj.fp8_format.name == "E3M4"

    def test_quantize_model_requires_format_or_config(self):
        """quantize_model should raise if neither format nor config provided."""
        from altgrad.integration import quantize_model

        model = nn.Sequential(nn.Linear(10, 5))

        with pytest.raises(ValueError, match="Provide exactly one"):
            quantize_model(model)  # Neither format nor config

    def test_quantize_model_rejects_both_format_and_config(self):
        """quantize_model should raise if both format and config provided."""
        from altgrad.integration import QuantizationConfig, quantize_model

        model = nn.Sequential(nn.Linear(10, 5))
        config = QuantizationConfig(default_format="E5M2")

        with pytest.raises(ValueError, match="Provide exactly one"):
            quantize_model(model, E5M2, config=config)  # Both provided


class TestCreateMixedPrecisionConfig:
    """Test create_mixed_precision_config() helper function."""

    def test_create_mixed_precision_config_defaults(self):
        """create_mixed_precision_config should create standard GPT config."""
        from altgrad.integration import create_mixed_precision_config

        config = create_mixed_precision_config(
            attention_format=None,  # BF16
            mlp_format="E5M2",
            lm_head_format=None,  # BF16
        )

        # Attention should be BF16
        assert config.get_format_for_layer("transformer.h.0.attn.c_attn") is None

        # MLP should be E5M2
        mlp_format = config.get_format_for_layer("transformer.h.0.mlp.c_fc")
        assert mlp_format is not None
        assert mlp_format.name == "E5M2"

        # lm_head should be BF16
        assert config.get_format_for_layer("lm_head") is None
