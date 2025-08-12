"""Tests for unified model configuration and seed diversification."""

import pytest
from unittest.mock import AsyncMock, patch
from src.models.model_config import (
    get_model_config,
    generate_unique_seeds,
    _detect_model_type
)
from src.models.litellm_model import LiteLLMModel


class TestModelConfiguration:
    """Test unified model configuration system."""

    def test_model_type_detection(self):
        """Test accurate model type detection."""
        assert _detect_model_type("gpt-4o") == "openai"
        assert _detect_model_type("claude-3-sonnet") == "claude"
        assert _detect_model_type("grok-beta") == "grok"
        assert _detect_model_type("gemini-pro") == "gemini"
        assert _detect_model_type("o1-preview") == "o_series"
        assert _detect_model_type("o3-mini") == "o_series"
        assert _detect_model_type("unknown-model") == "unknown"

    def test_model_config_properties(self):
        """Test model configurations have correct properties."""
        # Claude should require seed diversification
        claude_config = get_model_config("claude-3-sonnet")
        assert claude_config.requires_seed_diversification is True
        assert claude_config.requires_parallel_generation is True
        assert claude_config.max_concurrent == 15

        # O3 should require seed diversification
        o3_config = get_model_config("o3-mini")
        assert o3_config.requires_seed_diversification is True
        assert o3_config.requires_parallel_generation is True

        # OpenAI GPT should not require seed diversification
        gpt_config = get_model_config("gpt-4o")
        assert gpt_config.requires_seed_diversification is False
        assert gpt_config.requires_parallel_generation is False

    def test_seed_generation(self):
        """Test unique seed generation."""
        seeds = generate_unique_seeds(5)
        assert len(seeds) == 5
        assert len(set(seeds)) >= 4  # Should be mostly unique
        assert all(0 <= seed < 2**31 for seed in seeds)


class TestLiteLLMModelIntegration:
    """Test LiteLLM model with unified configuration."""

    @pytest.mark.asyncio
    async def test_claude_seed_diversification(self):
        """Test Claude model uses different seeds for parallel generation."""
        model = LiteLLMModel("claude-3-sonnet")

        # Verify configuration
        assert model.config.requires_seed_diversification is True

        with patch(
            'src.models.litellm_model.litellm.acompletion'
        ) as mock_completion:
            # Mock successful responses
            mock_completion.return_value = AsyncMock()
            mock_completion.return_value.choices = [
                AsyncMock(message=AsyncMock(content="response1")),
            ]

            # Should use parallel generation for n>1
            await model.generate_async("test prompt", n=3)

            # Verify multiple API calls were made (parallel generation)
            assert mock_completion.call_count == 3

            # Verify different seeds were used
            call_args = [call[1] for call in mock_completion.call_args_list]
            seeds_used = [args.get('seed') for args in call_args]
            # All seeds should be different
            assert len(set(seeds_used)) == 3

    def test_unified_async_config(self):
        """Test async config uses unified configuration."""
        claude_model = LiteLLMModel("claude-3-sonnet")
        config = claude_model.get_async_config()
        assert config.max_concurrent == 15
        assert config.rate_limit_delay == 0.1

        grok_model = LiteLLMModel("grok-beta")
        config = grok_model.get_async_config()
        assert config.max_concurrent == 10
        assert config.rate_limit_delay == 0.2
