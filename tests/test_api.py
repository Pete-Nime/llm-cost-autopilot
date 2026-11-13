
# Import FastAPI's test client.
# It lets us test endpoints without starting a web server.
from fastapi.testclient import TestClient

# Import our existing FastAPI application.
from src.api.main import app

# Import os to manage the execution environment during testing.
import os

# Import SimpleNamespace to create fake AI results.
from types import SimpleNamespace

# Import AsyncMock to simulate an asynchronous AI workflow.
from unittest.mock import AsyncMock

# Create a test client for our application.
client = TestClient(app)


# Test that the health endpoint works correctly.
def test_health_endpoint():

    # Send a simulated GET request to our API.
    response = client.get("/health")

    # Check that the server returns HTTP 200 (success).
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary.
    data = response.json()

    # Confirm the API reports a healthy status.
    assert data["status"] == "healthy"

    # Confirm the correct service name is returned.
    assert data["service"] == "LLM Cost Autopilot"


# Test that our analytics endpoint returns the expected fields.
def test_analytics_summary_endpoint():

    # Send a simulated request to the analytics endpoint.
    response = client.get("/analytics/summary")

    # Confirm the request succeeded.
    assert response.status_code == 200

    # Read the returned JSON.
    data = response.json()

    # Check that all expected analytics fields are present.
    expected_fields = [
        "total_attempts",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost_usd",
        "average_latency_ms",
        "fallback_attempts",
    ]

    # Check each expected field.
    for field in expected_fields:

        # Confirm the field exists in the JSON response.
        assert field in data


# Test that an unknown endpoint returns HTTP 404.
def test_unknown_endpoint():

    # Request a URL that does not exist.
    response = client.get("/this-route-does-not-exist")

    # Confirm FastAPI correctly reports "Not Found".
    assert response.status_code == 404


# Test that a simple question is routed correctly.
def test_route_preview_simple():

    # Send a simulated POST request to the routing API.
    response = client.post(
        "/route/preview",
        json={"prompt": "What is Python?"},
    )

    # Confirm the API request succeeded.
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary.
    data = response.json()

    # Confirm the router selected the expected model.
    assert data["selected_model"] == "gpt-4o-mini"

    # Confirm the prompt was classified as simple.
    assert data["complexity"] == "simple"

    # Confirm no paid AI request was executed.
    assert data["executed"] is False


# Test that an empty prompt is rejected.
def test_route_preview_empty_prompt():

    # Send an empty prompt to the API.
    response = client.post(
        "/route/preview",
        json={"prompt": ""},
    )

    # Confirm validation rejected the request.
    assert response.status_code == 422

# Test that whitespace-only prompts are rejected.
def test_route_preview_whitespace():

    # Send a prompt containing only spaces.
    response = client.post(
        "/route/preview",
        json={"prompt": "   "},
    )

    # Confirm the API rejects whitespace-only input.
    assert response.status_code == 422


# Test that excessively long prompts are rejected.
def test_route_preview_too_long():

    # Create a prompt containing 2001 characters.
    long_prompt = "A" * 2001

    # Send the oversized prompt to the API.
    response = client.post(
        "/route/preview",
        json={"prompt": long_prompt},
    )

    # Confirm the input length limit is enforced.
    assert response.status_code == 422

# Confirm paid AI execution is blocked by default.
def test_execution_disabled_by_default(monkeypatch):

    # Remove the environment flag for this test.
    monkeypatch.delenv("ENABLE_AI_EXECUTION", raising=False)

    # Submit a simulated execution request.
    response = client.post(
        "/route/execute",
        json={"prompt": "What is Python?"},
    )

    # Confirm execution was blocked.
    assert response.status_code == 403

    # Confirm the API returned the expected message.
    assert response.json()["detail"] == "AI execution is disabled."

# Test that our API adds the costs of both AI attempts.
def test_execution_reports_total_cost(monkeypatch):

    # Enable execution only inside this test.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

        # Simulate a successful budget check for any prompt.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        lambda prompt: True,
    )


    # Simulate the first model's response.
    first_response = SimpleNamespace(
        output_text="",
        model_id="gpt-4o-mini",
        cost_usd=0.0002,
    )

    # Simulate the fallback model's response.
    second_response = SimpleNamespace(
        output_text="Python is a programming language.",
        model_id="gpt-4o",
        cost_usd=0.0040,
    )

    # Create a fake workflow containing both responses.
    fake_result = SimpleNamespace(
        response=second_response,
        responses=(first_response, second_response),
        escalated=True,
        attempts=2,
    )

    # Replace the real AI workflow with a fake one.
    fake_workflow = AsyncMock(return_value=fake_result)

    # Prevent real provider calls and database writes.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Send a simulated API request.
    response = client.post(
        "/route/execute",
        json={"prompt": "What is Python?"},
    )

    # Confirm the API returned HTTP 200.
    assert response.status_code == 200

    # Read the JSON response.
    data = response.json()

    # Confirm the final model was GPT-4o.
    assert data["model"] == "gpt-4o"

    # Confirm escalation occurred.
    assert data["escalated"] is True

    # Confirm two attempts were made.
    assert data["attempts"] == 2

    # Confirm both estimated costs were added together.
    assert abs(data["estimated_cost_usd"] - 0.0042) < 1e-10

    # Confirm the total cost is below our $0.01 budget.
    assert data["budget_exceeded"] is False

    # Confirm the fake workflow was called once.
    fake_workflow.assert_awaited_once_with("What is Python?")


# Test that provider failures do not expose internal details.
def test_execution_handles_provider_failure(monkeypatch):

    # Enable execution for this test only.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

    # Simulate a successful budget check for any prompt.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        lambda prompt: True,
    )


    # Create a fake asynchronous workflow that fails.
    fake_workflow = AsyncMock(
        side_effect=RuntimeError(
            "Internal provider failure: secret test credential"
        )
    )

    # Replace the real workflow with the failing mock.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Submit a simulated API request.
    response = client.post(
        "/route/execute",
        json={"prompt": "Explain Python."},
    )

    # Confirm the API returns a controlled error.
    assert response.status_code == 503

    # Confirm the user receives a safe message.
    assert response.json()["detail"] == (
        "AI execution is temporarily unavailable."
    )

    # Confirm sensitive internal details are not returned.
    assert "secret test credential" not in response.text

    # Confirm the workflow was attempted once.
    fake_workflow.assert_awaited_once()


# Test that the API reports when a workflow exceeds its budget.
def test_execution_reports_budget_exceeded(monkeypatch):

    # Enable execution for this test only.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

        # Simulate a successful budget check for any prompt.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        lambda prompt: True,
    )
    # Create a simulated expensive AI response.
    expensive_response = SimpleNamespace(
        output_text="This is a simulated expensive answer.",
        model_id="gpt-4o",
        cost_usd=0.02,
    )

    # Create a fake workflow with one provider attempt.
    fake_result = SimpleNamespace(
        response=expensive_response,
        responses=(expensive_response,),
        escalated=False,
        attempts=1,
    )

    # Replace the real AI workflow with our fake workflow.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        AsyncMock(return_value=fake_result),
    )

    # Send a simulated request to FastAPI.
    response = client.post(
        "/route/execute",
        json={"prompt": "Explain distributed systems."},
    )

    # Confirm the API responded successfully.
    assert response.status_code == 200

    # Confirm the budget overrun was reported.
    assert response.json()["budget_exceeded"] is True


# Verify that over-budget requests never execute the AI workflow.
def test_budget_rejection_prevents_provider_call(monkeypatch):

    # Enable execution for this test only.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

        # Simulate a budget rejection for any prompt.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        lambda prompt: False,
    )

    # Create a fake AI workflow that records whether it was called.
    fake_workflow = AsyncMock()

    # Replace the real workflow with our fake one.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Send a simulated request to the API.
    response = client.post(
        "/route/execute",
        json={"prompt": "Explain distributed systems."},
    )

    # Confirm the API rejected the request.
    assert response.status_code == 403

    # Confirm the AI workflow was never called.
    fake_workflow.assert_not_awaited()



# Verify that FastAPI uses the real fallback-aware budget policy.
def test_api_uses_real_routed_budget_policy(monkeypatch):

    # Enable AI execution only during this test.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

    # Import our actual fallback-aware budget policy.
    from src.router.budget_policy import (
        is_routed_workflow_within_budget,
    )

    # Ensure FastAPI uses the real budget calculation.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        is_routed_workflow_within_budget,
    )

    # Create a fake response from GPT-4o-mini.
    # This does not contact OpenAI.
    fake_response = SimpleNamespace(
        output_text="Python is a programming language.",
        model_id="gpt-4o-mini",
        cost_usd=0.0002,
    )

    # Create a complete fake workflow result.
    fake_result = SimpleNamespace(
        response=fake_response,
        responses=(fake_response,),
        escalated=False,
        attempts=1,
    )

    # Make our fake workflow return the structured result.
    fake_workflow = AsyncMock(return_value=fake_result)

    # Replace the real AI workflow with our fake one.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Send a simple question through our API.
    response = client.post(
        "/route/execute",
        json={"prompt": "What is Python?"},
    )

    # Confirm the real budget policy allowed the request.
    assert response.status_code == 200

    # Confirm the selected model's response was returned.
    assert response.json()["model"] == "gpt-4o-mini"

    # Confirm the estimated cost was reported correctly.
    assert abs(
        response.json()["estimated_cost_usd"] - 0.0002
    ) < 1e-10

    # Confirm the request stayed within the budget.
    assert response.json()["budget_exceeded"] is False

    # Confirm our fake workflow ran exactly once.
    fake_workflow.assert_awaited_once_with("What is Python?")



# Verify that the API accepts a prompt at the character limit.
def test_preview_accepts_maximum_prompt_length():

    # Import our shared input-size configuration.
    from config.budget import MAX_PROMPT_CHARACTERS

    # Create a prompt exactly at the maximum length.
    prompt = "A" * MAX_PROMPT_CHARACTERS

    # Send the prompt to the preview endpoint.
    response = client.post(
        "/route/preview",
        json={"prompt": prompt},
    )

    # The prompt length itself should be valid.
    assert response.status_code == 200


# Verify that the execution endpoint rejects oversized prompts.
def test_execution_rejects_oversized_prompt(monkeypatch):

    # Enable AI execution for this test only.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

    # Import the shared input-size limit.
    from config.budget import MAX_PROMPT_CHARACTERS

    # Create a prompt that exceeds the limit.
    prompt = "A" * (MAX_PROMPT_CHARACTERS + 1)

    # Create a fake workflow to detect accidental execution.
    fake_workflow = AsyncMock()

    # Replace real AI execution with the fake workflow.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Submit the oversized prompt.
    response = client.post(
        "/route/execute",
        json={"prompt": prompt},
    )

    # FastAPI should reject the prompt before execution.
    assert response.status_code == 422

    # No provider workflow should run.
    fake_workflow.assert_not_awaited()


# Verify that execution rejects whitespace-only prompts.
def test_execution_rejects_whitespace(monkeypatch):

    # Enable execution for this test.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

    # Replace provider execution with a mock.
    fake_workflow = AsyncMock()

    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Send a prompt containing only spaces.
    response = client.post(
        "/route/execute",
        json={"prompt": "   "},
    )

    # The API must reject it.
    assert response.status_code == 422

    # The AI provider must never be contacted.
    fake_workflow.assert_not_awaited()



# Verify that FastAPI accepts V2 routing settings.
def test_route_preview_supports_v2(monkeypatch):

    # Import SimpleNamespace to create a fake routing decision.
    from types import SimpleNamespace

    # Create a fake decision from our V2 classifier.
    fake_decision = SimpleNamespace(
        complexity="complex",
        model_name="gpt-4o",
        reason="Selected using V2.",
    )

    # Create a function that checks the supplied settings.
    def fake_route_prompt(prompt, use_v2=False, risk_threshold=0.40):

        # Confirm the API passes the cleaned prompt.
        assert prompt == "Design a secure distributed system."

        # Confirm the V2 classifier was requested.
        assert use_v2 is True

        # Confirm the custom threshold was forwarded.
        assert risk_threshold == 0.35

        # Return the simulated routing decision.
        return fake_decision

    # Replace the real classifier to avoid model inference.
    monkeypatch.setattr(
        "src.api.main.route_prompt",
        fake_route_prompt,
    )

    # Submit a request with V2 enabled.
    response = client.post(
        "/route/preview",
        json={
            "prompt": "Design a secure distributed system.",
            "use_v2": True,
            "risk_threshold": 0.35,
        },
    )

    # Confirm the request succeeded.
    assert response.status_code == 200

    # Confirm the expected model was selected.
    assert response.json()["selected_model"] == "gpt-4o"

    # Confirm no AI provider request was made.
    assert response.json()["executed"] is False


# Test that V2 execution uses the same decision throughout.
def test_execution_uses_v2_decision(monkeypatch):

    # Enable execution only during this test.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

    # Create a simulated V2 routing decision.
    fake_decision = SimpleNamespace(
        complexity="complex",
        model_name="gpt-4o",
    )

    # Create a fake AI response.
    fake_response = SimpleNamespace(
        output_text="A secure system design.",
        model_id="gpt-4o",
        cost_usd=0.001,
    )

    # Create a fake completed workflow.
    fake_result = SimpleNamespace(
        response=fake_response,
        responses=(fake_response,),
        escalated=False,
        attempts=1,
    )

    # Replace the real classifier with our fake decision.
    def fake_route(prompt, use_v2=False, risk_threshold=0.40):

        # Confirm that V2 was requested.
        assert use_v2 is True

        # Return the predetermined decision.
        return fake_decision

    # Install the fake classifier.
    monkeypatch.setattr(
        "src.api.main.route_prompt",
        fake_route,
    )

    # Create a budget checker that verifies the selected decision.
    def fake_budget(prompt, decision=None):

        # Confirm budget checking uses the V2 decision.
        assert decision is fake_decision

        # Simulate an affordable request.
        return True

    # Install the fake budget checker.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        fake_budget,
    )

    # Replace real execution with an asynchronous mock.
    fake_workflow = AsyncMock(return_value=fake_result)

    # Prevent real provider calls and database writes.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Submit a V2 execution request.
    response = client.post(
        "/route/execute",
        json={
            "prompt": "Design a secure distributed system.",
            "use_v2": True,
            "risk_threshold": 0.40,
        },
    )

    # Confirm successful execution.
    assert response.status_code == 200

    # Confirm the expected model was reported.
    assert response.json()["model"] == "gpt-4o"

    # Confirm execution received the same decision.
    fake_workflow.assert_awaited_once_with(
        "Design a secure distributed system.",
        decision=fake_decision,
    )


# Verify that an over-budget V2 request never executes.
def test_v2_budget_rejection_blocks_execution(monkeypatch):

    # Enable execution only during this test.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

    # Create a simulated V2 decision.
    fake_decision = SimpleNamespace(
        complexity="complex",
        model_name="gpt-4o",
    )

    # Replace the real classifier.
    monkeypatch.setattr(
        "src.api.main.route_prompt",
        lambda prompt, use_v2, risk_threshold: fake_decision,
    )

    # Simulate a rejected budget check.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        lambda prompt, decision: False,
    )

    # Create a fake workflow to detect accidental execution.
    fake_workflow = AsyncMock()

    # Replace the real execution workflow.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Submit a V2 execution request.
    response = client.post(
        "/route/execute",
        json={
            "prompt": "Design a complex AI system.",
            "use_v2": True,
        },
    )

    # Confirm that FastAPI rejected the request.
    assert response.status_code == 403

    # Confirm no provider workflow was executed.
    fake_workflow.assert_not_awaited()


# Verify that FastAPI rejects invalid V2 risk thresholds.
def test_v2_rejects_invalid_risk_threshold():

    # Submit a threshold greater than the allowed maximum.
    response = client.post(
        "/route/preview",
        json={
            "prompt": "Explain distributed systems.",
            "use_v2": True,
            "risk_threshold": 1.5,
        },
    )

    # Confirm Pydantic rejected the invalid value.
    assert response.status_code == 422


# Verify that budget-estimation errors are handled safely.
def test_v2_budget_estimation_failure_returns_503(monkeypatch):

    # Enable AI execution for this isolated test.
    monkeypatch.setenv("ENABLE_AI_EXECUTION", "true")

    # Import a helper for creating a fake routing decision.
    from types import SimpleNamespace

    # Import a mock for our asynchronous execution workflow.
    from unittest.mock import AsyncMock

    # Create a fake V2 routing decision.
    fake_decision = SimpleNamespace(
        complexity="complex",
        model_name="gpt-4o",
    )

    # Replace the real V2 classifier with our fake decision.
    monkeypatch.setattr(
        "src.api.main.route_prompt",
        lambda prompt, use_v2, risk_threshold: fake_decision,
    )

    # Create a function that simulates a budget calculation failure.
    def failing_budget_check(prompt, decision=None):

        # Simulate an unexpected internal error.
        raise RuntimeError("Budget calculation failed")

    # Replace the real budget checker with the failing function.
    monkeypatch.setattr(
        "src.api.main.is_routed_workflow_within_budget",
        failing_budget_check,
    )

    # Create a fake workflow to detect accidental execution.
    fake_workflow = AsyncMock()

    # Prevent real AI provider requests.
    monkeypatch.setattr(
        "src.api.main.run_and_log",
        fake_workflow,
    )

    # Send a V2 execution request.
    response = client.post(
        "/route/execute",
        json={
            "prompt": "Design a secure AI system.",
            "use_v2": True,
        },
    )

    # Confirm FastAPI returns a controlled service error.
    assert response.status_code == 503

    # Confirm the internal exception message is not exposed.
    assert "Budget calculation failed" not in response.text

    # Confirm no AI execution took place.
    fake_workflow.assert_not_awaited()
