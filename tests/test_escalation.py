
# Import asyncio to run asynchronous tests.
import asyncio

# Import AsyncMock to simulate AI provider calls.
from unittest.mock import AsyncMock, patch

# Import the escalation function.
from src.verifier.escalation import route_with_escalation

# Import the standardized response structure.
from src.providers.response import ModelResponse

# Import pytest to run tests and check expected errors.
import pytest


# Create a fake response for testing without API charges.
def make_response(text: str, model_id: str) -> ModelResponse:

    # Return a standardized fake AI response.
    return ModelResponse(
        output_text=text,
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
        cost_usd=0.0001,
        model_id=model_id,
        provider="openai",
    )


# Confirm that a normal response does not trigger escalation.
def test_no_escalation_for_nonempty_answer():

    # Create a fake successful first response.
    first = make_response("Python is a language.", "gpt-4o-mini")

    # Replace the real provider function with a mock.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Configure the mock to return our first response.
        mock_send.return_value = first

        # Execute the escalation workflow.
        result = asyncio.run(
            route_with_escalation("What is Python?")
        )

        # Confirm that only one request was made.
        assert mock_send.await_count == 1

    # Confirm escalation did not occur.
    assert result.escalated is False
    assert result.attempts == 1


# Confirm that an empty response triggers one fallback request.
def test_escalation_for_empty_answer():

    # Simulate an empty first response.
    first = make_response("", "gpt-4o-mini")

    # Simulate a successful fallback response.
    second = make_response("Python is a language.", "gpt-4o")

    # Replace the real provider function with a mock.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Return the first and second responses in order.
        mock_send.side_effect = [first, second]

        # Execute the workflow.
        result = asyncio.run(
            route_with_escalation("What is Python?")
        )

        # Confirm exactly two requests were made.
        assert mock_send.await_count == 2

        # Confirm the second request used the fallback model.
        assert mock_send.await_args_list[1].kwargs[
            "model_config"
        ].model_id == "gpt-4o"

    # Confirm escalation occurred.
    assert result.escalated is True
    assert result.attempts == 2
    assert result.response.output_text == "Python is a language."


# Confirm we never escalate beyond the strongest configured model.
def test_no_escalation_from_strongest_model():

    # Simulate an empty response from gpt-4o.
    first = make_response("", "gpt-4o")

    # Replace the provider call with a mock.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Configure the empty response.
        mock_send.return_value = first

        # Execute a complex question routed to gpt-4o.
        result = asyncio.run(
            route_with_escalation("Design a fault-tolerant system.")
        )

        # Confirm no second request was made.
        assert mock_send.await_count == 1

    # Confirm escalation did not occur.
    assert result.escalated is False
    assert result.attempts == 1


# Check that escalation also returns the final quality report.
def test_escalation_returns_quality_result():

    # Simulate an empty answer from the first model.
    first = make_response("", "gpt-4o-mini")

    # Simulate a longer answer from the fallback model.
    second = make_response(
        "Python is a programming language used for "
        "automation, data analysis, machine learning, "
        "web development, and building software applications.",
        "gpt-4o",
    )

    # Replace real API requests with simulated responses.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Return the empty answer, followed by the fallback.
        mock_send.side_effect = [first, second]

        # Run the escalation workflow.
        result = asyncio.run(
            route_with_escalation("What is Python?")
        )

        # Confirm that exactly two AI requests were made.
        assert mock_send.await_count == 2

    # Confirm that escalation occurred.
    assert result.escalated is True

    # Confirm the total number of attempts.
    assert result.attempts == 2

    # Confirm the fallback answer passed our length threshold.
    assert result.quality.passed is True

    # Confirm the score is within the expected range.
    assert 0.0 <= result.quality.score <= 1.0



# Verify that an affordable fallback is allowed.
def test_fallback_allowed_when_budget_is_sufficient():

    # Simulate an empty first response with a small cost.
    first = make_response("", "gpt-4o-mini")

    # Simulate a successful GPT-4o fallback.
    second = make_response("Fallback succeeded.", "gpt-4o")

    # Replace real provider requests with simulated responses.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Return the first and second responses in order.
        mock_send.side_effect = [first, second]

        # Execute the escalation workflow.
        result = asyncio.run(
            route_with_escalation("What is Python?")
        )

        # Verify both provider requests were made.
        assert mock_send.await_count == 2

    # Confirm that fallback was allowed.
    assert result.escalated is True

    # Confirm the workflow used two attempts.
    assert result.attempts == 2


# Verify that an unaffordable fallback is blocked.
def test_fallback_blocked_when_budget_is_insufficient():

    # Simulate an empty first response.
    first = ModelResponse(
        output_text="",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,

        # Leave only $0.001 of the $0.01 budget.
        cost_usd=0.009,

        model_id="gpt-4o-mini",
        provider="openai",
    )

    # Replace real provider calls with a simulated response.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Return the expensive first response.
        mock_send.return_value = first

        # Execute the escalation workflow.
        result = asyncio.run(
            route_with_escalation("What is Python?")
        )

        # Verify that no second request was made.
        assert mock_send.await_count == 1

    # Confirm the fallback was blocked.
    assert result.escalated is False

    # Confirm only one provider attempt occurred.
    assert result.attempts == 1

    # Confirm the reason explains the budget restriction.
    assert "fallback blocked" in result.reason

    # Confirm only the first response is recorded.
    assert len(result.responses) == 1



# Verify invalid prompts never reach the AI provider.
def test_escalation_rejects_oversized_prompt():

    # Import the shared character limit.
    from config.budget import MAX_PROMPT_CHARACTERS

    # Create a prompt that exceeds the limit.
    oversized_prompt = "A" * (MAX_PROMPT_CHARACTERS + 1)

    # Replace real provider requests with a mock.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Run the workflow and capture its validation error.
        try:
            asyncio.run(route_with_escalation(oversized_prompt))

        # Confirm the expected error was raised.
        except ValueError:
            pass

        # Fail the test if no error occurred.
        else:
            raise AssertionError("Oversized prompt was not rejected")

        # Confirm no paid AI provider request was made.
        mock_send.assert_not_awaited()




# Verify an over-budget workflow is blocked before contacting AI.
@pytest.mark.asyncio
async def test_initial_budget_rejection_prevents_provider_call(
    monkeypatch,
):

    # Import the escalation module so we can modify
    # its budget value temporarily during this test.
    import src.verifier.escalation as escalation_module

    # Set a tiny test-only budget of $0.000001.
    # This does not change config/budget.py.
    monkeypatch.setattr(
        escalation_module,
        "MAX_REQUEST_COST_USD",
        0.000001,
    )

    # Use a valid question that normally routes to GPT-4o-mini.
    prompt = "What is Python?"

    # Replace the real AI provider with a fake function.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
    ) as mock_send:

        # Expect the workflow to reject the estimated cost.
        with pytest.raises(
            ValueError,
            match="Estimated workflow cost exceeds",
        ):
            await route_with_escalation(prompt)

        # Verify that no real or fake provider request was made.
        mock_send.assert_not_awaited()



# Verify that an affordable workflow can contact the AI provider.
@pytest.mark.asyncio
async def test_initial_budget_allows_affordable_workflow():

    # Create a short, affordable question.
    prompt = "What is Python?"

    # Simulate a successful AI provider response.
    fake_response = ModelResponse(
        output_text="Python is a programming language.",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
        cost_usd=0.0001,
        model_id="gpt-4o-mini",
        provider="openai",
    )

    # Replace the real provider with our simulated response.
    with patch(
        "src.verifier.escalation.send_request",
        new_callable=AsyncMock,
        return_value=fake_response,
    ) as mock_send:

        # Execute the workflow using the fake provider.
        result = await route_with_escalation(prompt)

        # Confirm the provider was contacted once.
        mock_send.assert_awaited_once()

        # Confirm the expected response was returned.
        assert result.response is fake_response

        # Confirm that escalation was unnecessary.
        assert result.escalated is False
