
# Import pytest for automated testing.
import pytest

# Import the router we just created.
from src.router.intelligent_router import route_prompt


# Test each difficulty level and its expected model.
@pytest.mark.parametrize(
    "prompt, expected_complexity, expected_model",
    [
        # A simple definition question.
        ("What is Python?", "simple", "gpt-4o-mini"),

        # An intermediate explanation question.
        (
            "Explain how an ETL pipeline works.",
            "medium",
            "claude-haiku",
        ),

        # A complex engineering question.
        (
            "Design a fault-tolerant system.",
            "complex",
            "gpt-4o",
        ),
    ],
)
def test_router_selects_expected_model(
    prompt,
    expected_complexity,
    expected_model,
):

    # Ask our router to choose a model.
    decision = route_prompt(prompt)

    # Check the complexity classification.
    assert decision.complexity == expected_complexity

    # Check that the correct model was selected.
    assert decision.model_name == expected_model

    # Confirm that the decision includes an explanation.
    assert decision.reason


# Test that an empty question is rejected.
def test_router_rejects_empty_prompt():

    # An empty question should raise ValueError.
    with pytest.raises(ValueError):
        route_prompt("")


# Verify that our router selects a model from the real registry.
def test_router_uses_registered_model():

    # Ask the router to classify a simple question.
    decision = route_prompt("What is Python?")

    # Check the selected registry name.
    assert decision.model_name == "gpt-4o-mini"

    # Check the provider from the loaded configuration.
    assert decision.model_config.provider == "openai"

    # Check the actual model identifier.
    assert decision.model_config.model_id == "gpt-4o-mini"

    # Check that the configuration includes a quality tier.
    assert decision.model_config.quality_tier == "low"


# Import asyncio to execute our asynchronous router.
import asyncio

# Import patch to replace the real API call during testing.
from unittest.mock import patch

# Import our new end-to-end routing function.
from src.router.intelligent_router import route_and_send

# Import the standard response structure.
from src.providers.response import ModelResponse


# Verify that routing connects correctly to the provider client.
def test_route_and_send_uses_selected_model():

    # Create a fake response so no API credits are used.
    fake_response = ModelResponse(
        output_text="Python is a programming language.",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
        cost_usd=0.0001,
        model_id="gpt-4o-mini",
        provider="openai",
    )

    # Create a fake asynchronous provider function.
    async def fake_send_request(prompt, model_config):

        # Confirm the original question reaches the client.
        assert prompt == "What is Python?"

        # Confirm the router selected the expected model.
        assert model_config.model_id == "gpt-4o-mini"

        # Return our fake AI answer.
        return fake_response

    # Replace the real provider function temporarily.
    with patch(
        "src.router.intelligent_router.send_request",
        side_effect=fake_send_request,
    ):

        # Execute the asynchronous routing function.
        result = asyncio.run(
            route_and_send("What is Python?")
        )

    # Confirm the routing decision.
    assert result.decision.model_name == "gpt-4o-mini"

    # Confirm the answer was returned correctly.
    assert result.response.output_text == (
        "Python is a programming language."
    )

    # Confirm the response includes token usage.
    assert result.response.input_tokens == 10
    assert result.response.output_tokens == 20
