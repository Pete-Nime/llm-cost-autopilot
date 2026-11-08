
# Import pytest for testing expected errors.
import pytest

# Import the token estimator we created.
from src.router.token_estimator import estimate_input_tokens

# Import our existing budget calculation.
from src.router.budget_policy import (
    estimate_routed_workflow_cost,
)


# Test a short question.
def test_short_prompt():

    # 15 characters -> 4 content tokens + 20 overhead.
    result = estimate_input_tokens("What is Python?")

    assert result == 24


# Test that longer prompts require more estimated tokens.
def test_long_prompt():

    # Create a prompt containing 10,000 characters.
    long_prompt = "A" * 10_000

    # Estimate the input tokens.
    result = estimate_input_tokens(long_prompt)

    # 10,000 / 4 + 20 = 2,520.
    assert result == 2520


# Test that whitespace-only prompts are rejected.
def test_empty_prompt():

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        estimate_input_tokens("   ")


# Test that non-string input is rejected.
def test_invalid_prompt_type():

    with pytest.raises(TypeError, match="Prompt must be a string"):
        estimate_input_tokens(123)


# Test that a longer prompt produces a larger cost estimate.
def test_longer_prompt_costs_more():

    # Calculate the short prompt's routed workflow estimate.
    short_cost = estimate_routed_workflow_cost(
        prompt="What is Python?",
    )

    # Calculate a longer simple-style prompt's estimate.
    long_cost = estimate_routed_workflow_cost(
        prompt="What is Python? " + ("A" * 1000),
    )

    # Both prompts should use the same initial model
    # for this test to compare input length fairly.
    assert long_cost > short_cost
