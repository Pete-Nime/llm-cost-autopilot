
# Import asyncio to execute asynchronous code.
import asyncio

# Import SQLite to inspect saved records.
import sqlite3

# Import Mock helpers to avoid real AI API calls.
from unittest.mock import AsyncMock, patch

# Import our workflow logging function.
from src.database.workflow_logger import run_and_log

# Import our existing result structures.
from src.verifier.escalation import EscalationResult
from src.verifier.quality_verifier import QualityResult

# Import the standardized AI response structure.
from src.providers.response import ModelResponse

import pytest
from types import SimpleNamespace



# Create a fake AI response for testing.
def make_response(model_id: str, cost: float) -> ModelResponse:

    # Return a simulated provider response.
    return ModelResponse(
        output_text="Python is a programming language.",
        input_tokens=10,
        output_tokens=20,
        latency_ms=150.0,
        cost_usd=cost,
        model_id=model_id,
        provider="openai",
    )


# Confirm both model attempts are saved.
def test_workflow_logs_both_attempts(tmp_path):

    # Create a temporary SQLite database.
    test_db = tmp_path / "workflow_test.db"

    # Simulate the initial model response.
    first = make_response("gpt-4o-mini", 0.0001)

    # Simulate the fallback model response.
    second = make_response("gpt-4o", 0.001)

    # Create a simulated escalation result.
    fake_result = EscalationResult(
        response=second,
        escalated=True,
        attempts=2,
        reason="Escalated once.",
        quality=QualityResult(
            passed=True,
            reason="Test response.",
            score=0.5,
        ),
        responses=(first, second),
    )

    # Replace the real AI workflow with our fake result.
    with patch(
        "src.database.workflow_logger.route_with_escalation",
        new_callable=AsyncMock,
    ) as mock_workflow:

        # Configure the fake workflow result.
        mock_workflow.return_value = fake_result

        # Run our logging wrapper.
        result = asyncio.run(
            run_and_log("What is Python?", db_path=test_db)
        )

    # Confirm we received the original workflow result.
    assert result is fake_result

    # Open the temporary SQLite database.
    with sqlite3.connect(test_db) as connection:

        # Read the saved model names and estimated costs.
        rows = connection.execute(
            """
            SELECT model_name, estimated_cost_usd, escalated
            FROM request_logs
            ORDER BY id
            """
        ).fetchall()

    # Confirm that both attempts were saved.
    assert len(rows) == 2

    # Confirm the first attempt used the cheaper model.
    assert rows[0][0] == "gpt-4o-mini"

    # Confirm the second attempt used the fallback model.
    assert rows[1][0] == "gpt-4o"

    # Confirm the escalation flags are correct.
    assert rows[0][2] == 0
    assert rows[1][2] == 1

    # Confirm the total estimated cost includes both attempts.
    assert abs(sum(row[1] for row in rows) - 0.0011) < 1e-10


# Verify that the logger uses the supplied V2 routing decision.
@pytest.mark.asyncio
async def test_workflow_logs_v2_selected_complexity(
    monkeypatch,
    tmp_path,
):

    # Create a fake routing decision made by V2.
    fake_decision = SimpleNamespace(
        complexity="complex",
        model_name="gpt-4o",
    )

    # Create a fake response instead of calling a paid AI provider.
    fake_response = SimpleNamespace(
        model_id="gpt-4o",
        provider="openai",
        input_tokens=20,
        output_tokens=40,
        latency_ms=500.0,
        cost_usd=0.001,
    )

    # Create a fake completed workflow.
    fake_result = SimpleNamespace(
        responses=(fake_response,),
    )

    # Replace the real escalation workflow with a fake one.
    fake_escalation = AsyncMock(return_value=fake_result)

    # Patch the function used inside our database logger.
    monkeypatch.setattr(
        "src.database.workflow_logger.route_with_escalation",
        fake_escalation,
    )

    # Create a fake database logger to capture the saved information.
    saved_records = []

    # Replace SQLite logging with an in-memory recording function.
    monkeypatch.setattr(
        "src.database.workflow_logger.log_request",
        lambda **kwargs: saved_records.append(kwargs),
    )

    # Execute our logging workflow using the V2 decision.
    await run_and_log(
        "Design a secure distributed authentication system.",
        db_path=tmp_path / "test_v2.db",
        decision=fake_decision,
    )

    # Confirm the selected decision was passed to escalation.
    fake_escalation.assert_awaited_once_with(
        "Design a secure distributed authentication system.",
        decision=fake_decision,
    )

    # Confirm exactly one provider attempt was recorded.
    assert len(saved_records) == 1

    # Confirm SQLite would receive the V2-selected complexity.
    assert saved_records[0]["complexity"] == "complex"

    # Confirm the correct model was recorded.
    assert saved_records[0]["model_name"] == "gpt-4o"

