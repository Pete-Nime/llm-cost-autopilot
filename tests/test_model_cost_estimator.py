
# Import pytest to test expected errors.
import pytest

# Import the new model-specific estimator.
from src.router.model_cost_estimator import estimate_model_cost


# Check the configured price for GPT-4o-mini.
def test_estimate_gpt_4o_mini_cost():

    # Simulate 1,000 input tokens and 500 output tokens.
    cost = estimate_model_cost(
        model_name="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=500,
    )

    # Input: 1000 * $0.15 / 1,000,000 = $0.00015
    # Output: 500 * $0.60 / 1,000,000 = $0.00030
    # Total: $0.00045
    assert cost == pytest.approx(0.00045)


# Check the configured price for GPT-4o.
def test_estimate_gpt_4o_cost():

    # Use the same token counts for a fair comparison.
    cost = estimate_model_cost(
        model_name="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
    )

    # Total: $0.0025 + $0.005 = $0.0075
    assert cost == pytest.approx(0.0075)


# Check that a local Ollama model has no per-token API charge.
def test_estimate_ollama_cost():

    cost = estimate_model_cost(
        model_name="llama3.2",
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost == 0.0


# Check that an unknown model is rejected.
def test_unknown_model():

    with pytest.raises(ValueError, match="Unknown model"):
        estimate_model_cost(
            model_name="not-a-real-model",
            input_tokens=100,
            output_tokens=50,
        )


# Check that negative token counts are rejected.
def test_negative_token_count():

    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        estimate_model_cost(
            model_name="gpt-4o-mini",
            input_tokens=-100,
            output_tokens=50,
        )
