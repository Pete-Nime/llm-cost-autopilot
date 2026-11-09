
# Import pytest to help us test our Python code.
import pytest

# Import the function that sends requests to AI providers.
from src.providers.client import send_request

# Import the class that stores AI model information.
from src.providers.model_registry import ModelConfig

# Import pytest to mark our asynchronous tests.
import pytest

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



# Import SimpleNamespace to create fake provider responses.
from types import SimpleNamespace

# Import AsyncMock to simulate asynchronous provider methods.
from unittest.mock import AsyncMock

# Import our shared output-token limit.
from config.budget import MAX_OUTPUT_TOKENS

# Import the model registry and provider client.
from src.providers.model_registry import load_model_registry
from src.providers.client import send_request

# Tell pytest to run this test using an asyncio event loop.
@pytest.mark.asyncio

# Test that OpenAI receives the configured token limit.
async def test_openai_output_token_limit(monkeypatch):

    # Provide a fake key so no real credentials are needed.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Create a simulated OpenAI API response.
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello!")
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
        ),
    )

    # Create a fake asynchronous OpenAI request.
    fake_create = AsyncMock(return_value=fake_response)

    # Create a fake OpenAI client.
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fake_create)
        )
    )

    # Replace the real OpenAI SDK client with our fake.
    monkeypatch.setattr(
        "src.providers.client.AsyncOpenAI",
        lambda api_key: fake_client,
    )

    # Load our existing model registry.
    registry = load_model_registry()

    # Call our provider interface using the fake client.
    await send_request("Say hello", registry["gpt-4o-mini"])

    # Confirm the request used our configured output limit.
    assert fake_create.await_args.kwargs[
        "max_completion_tokens"
    ] == MAX_OUTPUT_TOKENS

# Tell pytest to run this test using an asyncio event loop.
@pytest.mark.asyncio

# Test that Anthropic receives the configured token limit.
async def test_anthropic_output_token_limit(monkeypatch):

    # Provide a fake Anthropic key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Create a simulated Anthropic response.
    fake_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Hello!")
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
        ),
    )

    # Create a fake asynchronous Anthropic request.
    fake_create = AsyncMock(return_value=fake_response)

    # Create a fake Anthropic client.
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=fake_create)
    )

    # Replace the real Anthropic SDK client.
    monkeypatch.setattr(
        "src.providers.client.AsyncAnthropic",
        lambda api_key: fake_client,
    )

    # Load the model registry.
    registry = load_model_registry()

    # Call the provider interface using our fake.
    await send_request("Say hello", registry["claude-haiku"])

    # Confirm Anthropic received our output-token limit.
    assert fake_create.await_args.kwargs[
        "max_tokens"
    ] == MAX_OUTPUT_TOKENS

# Tell pytest to run this test using an asyncio event loop.
@pytest.mark.asyncio

# Test that Ollama receives the configured token limit.
async def test_ollama_output_token_limit(monkeypatch):

    # Create a fake Ollama HTTP response.
    fake_response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "response": "Hello!",
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
    )

    # Create a fake HTTP client.
    fake_http_client = SimpleNamespace(
        post=AsyncMock(return_value=fake_response),
    )

    # Create a fake asynchronous context manager.
    class FakeAsyncClient:

        # Return our fake HTTP client when entering.
        async def __aenter__(self):
            return fake_http_client

        # Exit without suppressing exceptions.
        async def __aexit__(self, exc_type, exc, tb):
            return False

    # Replace the real HTTP client with our fake.
    monkeypatch.setattr(
        "src.providers.client.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(),
    )

    # Load our existing registry.
    registry = load_model_registry()

    # Call the local-model provider interface.
    await send_request("Say hello", registry["llama3.2"])

    # Read the JSON body sent to the fake Ollama server.
    sent_json = fake_http_client.post.await_args.kwargs["json"]

    # Confirm Ollama received our configured generation limit.
    assert sent_json["options"]["num_predict"] == MAX_OUTPUT_TOKENS



# Verify direct provider calls reject oversized prompts.
@pytest.mark.asyncio
async def test_provider_rejects_oversized_prompt(monkeypatch):

    # Import our shared character limit.
    from config.budget import MAX_PROMPT_CHARACTERS

    # Load an existing model configuration.
    registry = load_model_registry()

    # Create a prompt one character over the limit.
    prompt = "A" * (MAX_PROMPT_CHARACTERS + 1)

    # Create a fake provider client that must not be used.
    fake_client = AsyncMock()

    # Replace the real OpenAI client with our fake.
    monkeypatch.setattr(
        "src.providers.client.AsyncOpenAI",
        fake_client,
    )

    # Expect the validator to reject the oversized prompt.
    with pytest.raises(ValueError):
        await send_request(prompt, registry["gpt-4o-mini"])

    # Confirm no provider client was created.
    fake_client.assert_not_called()


# Verify direct provider calls reject non-string inputs.
@pytest.mark.asyncio
async def test_provider_rejects_invalid_prompt_type():

    # Load an existing registered model.
    registry = load_model_registry()

    # Passing an integer must raise TypeError.
    with pytest.raises(TypeError):
        await send_request(123, registry["gpt-4o-mini"])


# Verify the provider receives the cleaned prompt.
@pytest.mark.asyncio
async def test_provider_receives_cleaned_prompt(monkeypatch):

    # Use a fake API key for this test.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Create a fake provider response.
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Hello!")
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
        ),
    )

    # Create a fake asynchronous provider request.
    fake_create = AsyncMock(return_value=fake_response)

    # Replace the real OpenAI SDK client.
    monkeypatch.setattr(
        "src.providers.client.AsyncOpenAI",
        lambda api_key: SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=fake_create
                )
            )
        ),
    )

    # Load our existing model registry.
    registry = load_model_registry()

    # Submit a prompt with surrounding whitespace.
    await send_request(
        "  Say hello  ",
        registry["gpt-4o-mini"],
    )

    # Inspect the messages sent to the fake provider.
    sent_messages = fake_create.await_args.kwargs["messages"]

    # Confirm the provider received the cleaned text.
    assert sent_messages == [
        {"role": "user", "content": "Say hello"}
    ]
