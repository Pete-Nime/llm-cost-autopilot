
# Import pytest to run our automated test.
import pytest

# Import SimpleNamespace to create a simple fake response.
from types import SimpleNamespace

# Import the benchmark function we want to test.
from src.providers.benchmark import benchmark_model


# Tell pytest this test uses asynchronous code.
@pytest.mark.asyncio
async def test_benchmark_model(monkeypatch):

    # Create a fake AI response with known values.
    fake_response = SimpleNamespace(
        provider="openai",
        model_id="fake-model",
        output_text="Artificial intelligence is machine intelligence.",
        input_tokens=10,
        output_tokens=8,
        latency_ms=150.0,
        cost_usd=0.00001
    )

    # Create a fake provider function.
    async def fake_send_request(prompt, model_config):

        # Return our fake response instead of calling an API.
        return fake_response

    # Replace the real provider function inside benchmark.py.
    monkeypatch.setattr(
        "src.providers.benchmark.send_request",
        fake_send_request
    )

    # Run the benchmark using one fake question.
    results = await benchmark_model(
        "fake-model",
        None,
        ["Explain AI."]
    )

    # Confirm exactly one result was returned.
    assert len(results) == 1

    # Confirm the request succeeded.
    assert results[0]["status"] == "success"

    # Confirm the fake answer was recorded correctly.
    assert results[0]["output_text"] == fake_response.output_text

    # Confirm the cost was recorded correctly.
    assert results[0]["cost_usd"] == fake_response.cost_usd
