
# Import pytest to help us test our Python code.
import pytest

# Import the function that sends requests to AI providers.
from src.providers.client import send_request

# Import the class that stores AI model information.
from src.providers.model_registry import ModelConfig


# Tell pytest that this test uses asynchronous Python.
@pytest.mark.asyncio
async def test_unsupported_provider():

    # Create a fake AI model for testing.
    # We deliberately use a provider that does not exist.
    model = ModelConfig(

        # Name of our fake provider.
        provider="unknown",

        # Name of our fake model.
        model_id="fake-model",

        # Example cost for one million input tokens.
        input_cost_per_million=1.0,

        # Example cost for one million output tokens.
        output_cost_per_million=2.0,

        # Example response time in milliseconds.
        average_latency_ms=500,

        # Assign the model to the low-quality tier.
        quality_tier="low"
    )

    # Check that our system raises the expected error.
    with pytest.raises(
        ValueError,
        match="Unsupported provider"
    ):

        # Attempt to send a prompt to our fake provider.
        # Our system should reject this request.
        await send_request(
            prompt="Hello",
            model_config=model
        )
