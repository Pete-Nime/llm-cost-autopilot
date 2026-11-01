
import pytest

from src.providers.model_registry import ModelConfig


def test_model_cost_calculation():
    """
    Check that the model calculates costs correctly.
    """

    # These are example prices, not live provider prices.
    model = ModelConfig(
        provider="example",
        model_id="test-model",
        input_cost_per_million=1.00,
        output_cost_per_million=4.00,
        average_latency_ms=500,
        quality_tier="low"
    )

    # 1,000 input tokens + 500 output tokens
    cost = model.calculate_cost(
        input_tokens=1000,
        output_tokens=500
    )

    # Input: 0.001 USD
    # Output: 0.002 USD
    # Total: 0.003 USD
    assert cost == pytest.approx(0.003)


def test_negative_token_count():
    """
    Negative token counts must be rejected.
    """

    model = ModelConfig(
        provider="example",
        model_id="test-model",
        input_cost_per_million=1.00,
        output_cost_per_million=4.00,
        average_latency_ms=500,
        quality_tier="low"
    )

    with pytest.raises(ValueError):
        model.calculate_cost(-1, 500)



# Import the function that loads our model registry.
from src.providers.model_registry import load_model_registry


# Check that our model registry loads correctly.
def test_load_model_registry():

    # Load the models from config/models.yaml.
    registry = load_model_registry()

    # Confirm GPT-4o exists in the registry.
    assert "gpt-4o" in registry

    # Confirm GPT-4o-mini exists in the registry.
    assert "gpt-4o-mini" in registry

    # Confirm Anthropic Sonnet exists.
    assert "claude-sonnet" in registry

    # Confirm Anthropic Haiku exists.
    assert "claude-haiku" in registry

    # Confirm our local Ollama model exists.
    assert "llama3.2" in registry

    # Confirm GPT-4o belongs to OpenAI.
    assert registry["gpt-4o"].provider == "openai"

    # Confirm the local model belongs to Ollama.
    assert registry["llama3.2"].provider == "ollama"

    # Confirm five models were loaded.
    assert len(registry) == 5
