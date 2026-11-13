
# Import pytest for automated testing.
import pytest

# Import the router we just created.
from src.router.intelligent_router import route_prompt

from unittest.mock import AsyncMock, patch
from src.router.intelligent_router import route_and_send


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



# Verify the router rejects oversized prompts.
def test_router_rejects_oversized_prompt():

    # Import our shared character limit.
    from config.budget import MAX_PROMPT_CHARACTERS

    # Create a prompt longer than our configured limit.
    prompt = "A" * (MAX_PROMPT_CHARACTERS + 1)

    # Import pytest locally so no new global import is needed.
    import pytest

    # The router must reject the oversized prompt.
    with pytest.raises(ValueError):
        route_prompt(prompt)


# Verify the direct routing workflow rejects oversized prompts.
def test_route_and_send_rejects_oversized_prompt():

    # Import asyncio to run our asynchronous function.
    import asyncio

    # Import the shared character limit.
    from config.budget import MAX_PROMPT_CHARACTERS

    # Import pytest for exception testing.
    import pytest

    # Import our direct routing function.
    from src.router.intelligent_router import route_and_send

    # Create an oversized prompt.
    prompt = "A" * (MAX_PROMPT_CHARACTERS + 1)

    # The workflow must fail before reaching an AI provider.
    with pytest.raises(ValueError):
        asyncio.run(route_and_send(prompt))



# Import SimpleNamespace to create fake V2 predictions.
from types import SimpleNamespace


# Verify that enabling V2 uses its predicted complexity.
def test_router_uses_v2_when_enabled():

    # Create a fake medium prediction with low complex probability.
    fake_prediction = SimpleNamespace(
        complexity="medium",
        probabilities={
            "simple": 0.10,
            "medium": 0.80,
            "complex": 0.10,
        },
    )

    # Replace the real V2 classifier temporarily.
    with patch(
        "src.router.intelligent_router.predict_v2_complexity",
        return_value=fake_prediction,
    ) as mock_predict:

        # Enable V2 explicitly.
        decision = route_prompt(
            "Explain database indexes.",
            use_v2=True,
        )

    # Confirm the V2 classifier was called once.
    mock_predict.assert_called_once()

    # Confirm the medium route was selected.
    assert decision.complexity == "medium"

    # Confirm the registered medium model was selected.
    assert decision.model_name == "claude-haiku"


# Verify that the risk policy escalates uncertain medium tasks.
def test_router_v2_escalates_medium_to_complex():

    # Create a medium prediction with 45% complex probability.
    fake_prediction = SimpleNamespace(
        complexity="medium",
        probabilities={
            "simple": 0.05,
            "medium": 0.50,
            "complex": 0.45,
        },
    )

    # Replace the real V2 classifier temporarily.
    with patch(
        "src.router.intelligent_router.predict_v2_complexity",
        return_value=fake_prediction,
    ):

        # Enable V2 with a 40% escalation threshold.
        decision = route_prompt(
            "Analyze a distributed transaction failure.",
            use_v2=True,
            risk_threshold=0.40,
        )

    # Confirm the safety policy escalated the route.
    assert decision.complexity == "complex"

    # Confirm the stronger configured model was selected.
    assert decision.model_name == "gpt-4o"


# Verify that the original classifier remains the default.
def test_router_original_mode_remains_default():

    # Prevent the V2 classifier from being used.
    with patch(
        "src.router.intelligent_router.predict_v2_complexity",
    ) as mock_predict:

        # Call the router without enabling V2.
        decision = route_prompt("What is Python?")

    # Confirm the original classifier handled the prompt.
    mock_predict.assert_not_called()

    # Confirm the original model selection is unchanged.
    assert decision.model_name == "gpt-4o-mini"


# Verify that invalid thresholds are rejected in V2 mode.
def test_router_v2_rejects_invalid_threshold():

    # Create a fake prediction for the routing policy.
    fake_prediction = SimpleNamespace(
        complexity="medium",
        probabilities={"complex": 0.50},
    )

    # Replace the real classifier during this test.
    with patch(
        "src.router.intelligent_router.predict_v2_complexity",
        return_value=fake_prediction,
    ):

        # A threshold above 100% must be rejected.
        with pytest.raises(ValueError):

            # Attempt to route with an invalid threshold.
            route_prompt(
                "Explain database replication.",
                use_v2=True,
                risk_threshold=1.50,
            )



# Verify that route_and_send passes V2 settings to the router.
@pytest.mark.asyncio
async def test_route_and_send_supports_v2():

    # Create a fake routing decision using our existing router.
    fake_decision = route_prompt("What is Python?")

    # Replace the routing function during this test.
    with patch(
        "src.router.intelligent_router.route_prompt",
        return_value=fake_decision,
    ) as mock_route:

        # Replace the real provider request with a fake response.
        with patch(
            "src.router.intelligent_router.send_request",
            new_callable=AsyncMock,
        ) as mock_send:

            # Call the request workflow with V2 enabled.
            result = await route_and_send(
                "Explain database indexing.",
                use_v2=True,
                risk_threshold=0.40,
            )

    # Verify that the router received the correct settings.
    mock_route.assert_called_once_with(
        "Explain database indexing.",
        use_v2=True,
        risk_threshold=0.40,
    )

    # Verify that the provider was called once.
    mock_send.assert_awaited_once()

    # Verify that the decision was returned.
    assert result.decision == fake_decision

# Verify that default requests still use the original mode.
@pytest.mark.asyncio
async def test_route_and_send_preserves_default_mode():

    # Create a fake routing decision.
    fake_decision = route_prompt("What is Python?")

    # Replace the routing function during this test.
    with patch(
        "src.router.intelligent_router.route_prompt",
        return_value=fake_decision,
    ) as mock_route:

        # Replace the real provider request with a fake response.
        with patch(
            "src.router.intelligent_router.send_request",
            new_callable=AsyncMock,
        ):

            # Call the workflow without enabling V2.
            await route_and_send("What is Python?")

    # Verify that the original classifier remains the default.
    mock_route.assert_called_once_with(
        "What is Python?",
        use_v2=False,
        risk_threshold=0.40,
    )
