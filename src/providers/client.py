
# Import asyncio to run blocking work asynchronously.
import asyncio

# Import time to measure request duration.
import time

# Import os to read secret environment variables.
import os

# Import the OpenAI SDK.
from openai import AsyncOpenAI

# Import the Anthropic SDK.
from anthropic import AsyncAnthropic

# Import httpx to communicate with local Ollama.
import httpx

# Import our existing model configuration.
from src.providers.model_registry import ModelConfig

# Import our standardized response class.
from src.providers.response import ModelResponse

# Import the maximum number of output tokens allowed.
# This value is shared by all three AI providers.
from config.budget import MAX_OUTPUT_TOKENS

# Import our shared input validator.
from src.router.prompt_validator import validate_prompt

# Define the one function used to call every provider.
async def send_request(
    prompt: str,
    model_config: ModelConfig
) -> ModelResponse:

    # Validate the prompt before contacting any AI provider.
    # Reject invalid types, blank prompts and oversized inputs.
    prompt = validate_prompt(prompt)

    # Remember when the request started.
    start_time = time.perf_counter()

    # Check which provider was selected.
    if model_config.provider == "openai":

        # Read the OpenAI API key from the environment.
        api_key = os.getenv("OPENAI_API_KEY")

        # Stop if the API key is missing.
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing")

        # Create an asynchronous OpenAI client.
        client = AsyncOpenAI(api_key=api_key)

        # Send the prompt to OpenAI with an output-token limit.
        response = await client.chat.completions.create(

            # Use the model selected by our intelligent router.
            model=model_config.model_id,

            # Send the customer's question.
            messages=[
                {"role": "user", "content": prompt}
            ],

            # Limit the maximum number of completion tokens.
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )

        # Extract the text returned by the model.
        output_text = response.choices[0].message.content or ""

        # Read actual input and output token usage.
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

    # Handle requests sent to Anthropic.
    elif model_config.provider == "anthropic":

        # Read the Anthropic API key.
        api_key = os.getenv("ANTHROPIC_API_KEY")

        # Stop if the API key is missing.
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing")

        # Create an asynchronous Anthropic client.
        client = AsyncAnthropic(api_key=api_key)

        # Send the prompt using Anthropic's Messages API.
        response = await client.messages.create(
            model=model_config.model_id,
            # Use the same configured output-token limit.
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Combine any returned text blocks.
        output_text = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        )

        # Read Anthropic's token usage.
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    # Handle requests to a locally running Ollama server.
    elif model_config.provider == "ollama":

        # Read the local Ollama server address.
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )

        # Open an asynchronous HTTP client.
        async with httpx.AsyncClient(timeout=120.0) as client:

            # Send the prompt to Ollama's generation endpoint.
            response = await client.post(
                f"{base_url}/api/generate",
                json={

                    # Select the locally installed model.
                    "model": model_config.model_id,

                    # Send the customer's question.
                    "prompt": prompt,

                    # Return the complete response instead of streaming.
                    "stream": False,

                    # Configure the maximum number of generated tokens.
                    "options": {
                        "num_predict": MAX_OUTPUT_TOKENS,
                    },
                }
            )

            # Raise an error for unsuccessful HTTP responses.
            response.raise_for_status()

            # Convert the JSON response into a Python dictionary.
            result = response.json()

        # Extract the generated text.
        output_text = result["response"]

        # Extract token counts supplied by Ollama.
        input_tokens = result["prompt_eval_count"]
        output_tokens = result["eval_count"]

    # Reject providers that our system does not support.
    else:
        raise ValueError(
            f"Unsupported provider: {model_config.provider}"
        )

    # Remember when the request finished.
    end_time = time.perf_counter()

    # Convert the elapsed time from seconds to milliseconds.
    latency_ms = (end_time - start_time) * 1000

    # Calculate the estimated price using our registry.
    cost_usd = model_config.calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    # Return the same response structure for every provider.
    return ModelResponse(
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model_id=model_config.model_id,
        provider=model_config.provider
    )
