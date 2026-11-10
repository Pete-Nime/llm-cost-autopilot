# Import pytest to check expected errors.
import pytest


from src.router.budget_policy import (
    estimate_max_workflow_cost,
    is_workflow_within_budget,
    estimate_model_workflow_cost,
    is_model_within_budget,
    estimate_routed_workflow_cost,
    is_routed_workflow_within_budget,
)


# Verify our planning estimate is calculated correctly.
def test_estimated_workflow_cost():

    # Expected cost with the current example configuration.
    expected_cost = 0.019

    # Compare floating-point numbers with a small tolerance.
    assert abs(estimate_max_workflow_cost() - expected_cost) < 1e-10


# Verify the configured budget rejects this estimate.
def test_workflow_exceeds_budget():

    # The estimated cost is above our $0.01 threshold.
    assert is_workflow_within_budget() is False



# Check the cost estimate for two GPT-4o-mini attempts.
def test_model_workflow_cost():

    # Each attempt uses 1,000 input and 300 output tokens.
    cost = estimate_model_workflow_cost(
        model_name="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=300,
        attempts=2,
    )

    # One attempt costs $0.00033.
    # Two attempts cost $0.00066.
    assert abs(cost - 0.00066) < 1e-10


# Verify a low-cost model fits our configured budget.
def test_low_cost_model_within_budget():

    result = is_model_within_budget(
        model_name="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=300,
        attempts=2,
    )

    assert result is True


# Verify an expensive model exceeds our configured budget.
def test_expensive_model_exceeds_budget():

    result = is_model_within_budget(
        model_name="gpt-4o",
        input_tokens=1000,
        output_tokens=1000,
        attempts=2,
    )

    # Two attempts cost $0.025, exceeding our $0.01 budget.
    assert result is False


# Verify invalid attempt counts are rejected.
def test_invalid_attempt_count():

    import pytest

    with pytest.raises(ValueError, match="Attempts must be at least 1"):
        estimate_model_workflow_cost(
            model_name="gpt-4o-mini",
            attempts=0,
        )



# Test the cost of a simple question with possible fallback.
def test_simple_routed_workflow_cost():

    # A simple question routes to GPT-4o-mini first.
    cost = estimate_routed_workflow_cost(
        prompt="What is Python?",
        input_tokens=1000,
        output_tokens=300,
    )

    # GPT-4o-mini: $0.00033
    # GPT-4o fallback: $0.00550
    # Combined: $0.00583
    assert abs(cost - 0.00583) < 1e-10


# Test that complex questions need only one GPT-4o call.
def test_complex_routed_workflow_cost():

    cost = estimate_routed_workflow_cost(
        prompt="Design a fault-tolerant distributed system.",
        input_tokens=1000,
        output_tokens=300,
    )

    # GPT-4o is already the fallback model.
    assert abs(cost - 0.0055) < 1e-10


# Test that a simple workflow fits the configured budget.
def test_simple_routed_workflow_within_budget():

    result = is_routed_workflow_within_budget(
        prompt="What is Python?",
        input_tokens=1000,
        output_tokens=300,
    )

    assert result is True


# Test that a larger possible workflow is rejected.
def test_large_routed_workflow_exceeds_budget():

    result = is_routed_workflow_within_budget(
        prompt="What is Python?",
        input_tokens=1000,
        output_tokens=1000,
    )

    # The estimated cost of both attempts exceeds $0.01.
    assert result is False



# Reject a negative output-token count.
def test_negative_output_tokens():

    # Negative token counts are not valid.
    with pytest.raises(ValueError):
        estimate_routed_workflow_cost(
            prompt="What is Python?",
            input_tokens=100,
            output_tokens=-1,
        )


# Reject fractional provider attempts.
def test_fractional_attempt_count():

    # We cannot make half of an AI request.
    with pytest.raises(TypeError):
        estimate_model_workflow_cost(
            model_name="gpt-4o-mini",
            attempts=1.5,
        )


# Reject fractional input-token counts.
def test_fractional_input_tokens():

    # Token counts must be whole numbers.
    with pytest.raises(TypeError):
        estimate_model_workflow_cost(
            model_name="gpt-4o-mini",
            input_tokens=12.5,
        )


# Reject boolean token counts.
def test_boolean_token_count():

    # Python treats True like 1, but this is not a valid token count.
    with pytest.raises(TypeError):
        estimate_routed_workflow_cost(
            prompt="What is Python?",
            input_tokens=True,
        )


# Confirm zero output tokens are allowed by the estimator.
def test_zero_output_token_estimate():

    # This checks the estimator, not provider API requirements.
    cost = estimate_model_workflow_cost(
        model_name="gpt-4o-mini",
        input_tokens=100,
        output_tokens=0,
        attempts=1,
    )

    # Input tokens still have a cost.
    assert cost >= 0
