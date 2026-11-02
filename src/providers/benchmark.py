
# Import asyncio to run our asynchronous benchmark.
import asyncio

# Import json to save results in a readable file.
import json

# Import os to read environment settings.
import os

# Import Path to create file paths safely.
from pathlib import Path

# Import load_dotenv to load our private API keys.
from dotenv import load_dotenv

# Import our existing provider request function.
from src.providers.client import send_request

# Import the existing model registry loader.
from src.providers.model_registry import load_model_registry

# Place this alongside the other import statements.
from datetime import datetime, timezone


# Find the project's root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load private environment variables from the project .env file.
load_dotenv(PROJECT_ROOT / ".env")

# Create our list of 10 benchmark questions.
PROMPTS = [
    "Explain artificial intelligence in one sentence.",
    "What is the difference between AI and machine learning?",
    "Summarize the benefits of cloud computing in two sentences.",
    "Write a Python function that adds two numbers.",
    "Explain what a SQL JOIN does.",
    "What is an API?",
    "Explain the purpose of unit testing.",
    "Describe how an ETL pipeline works.",
    "What is retrieval-augmented generation?",
    "Explain why data privacy matters in AI systems."
]


# Define a function that benchmarks one model.
async def benchmark_model(model_name, model_config, prompts):

    # Create an empty list to store this model's results.
    results = []

    # Process each benchmark question.
    for prompt_number, prompt in enumerate(prompts, start=1):

        # Show which question is being tested.
        print(f"Testing {model_name}: prompt {prompt_number}/{len(prompts)}")

        try:
            # Send the question to the selected AI provider.
            response = await send_request(prompt, model_config)

            # Record the successful response.
            results.append({
                "model": model_name,
                "provider": response.provider,
                "model_id": response.model_id,
                "prompt_number": prompt_number,
                "prompt": prompt,
                "output_text": response.output_text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency_ms": response.latency_ms,
                "cost_usd": response.cost_usd,
                "status": "success"
            })

        except Exception as error:
            # Record an error without stopping the whole benchmark.
            results.append({
                "model": model_name,
                "prompt_number": prompt_number,
                "prompt": prompt,
                "status": "error",
                "error": str(error)
            })

            # Show the error in the terminal.
            print(f"Error from {model_name}: {error}")

    # Return all results for this model.
    return results


# Define the main benchmark function.
async def main():

    # Load our configured models from YAML.
    registry = load_model_registry(
        PROJECT_ROOT / "config" / "models.yaml"
    )

    # Read the model selection from an environment variable.
    # By default, use only gpt-4o-mini to control API spending.
    selected_names = os.getenv(
        "BENCHMARK_MODELS",
        "gpt-4o-mini"
    ).split(",")

    # Create an empty list for every model's results.
    all_results = []

    # Run the benchmark for each selected model.
    for raw_name in selected_names:

        # Remove accidental spaces around model names.
        model_name = raw_name.strip()

        # Reject unknown model names.
        if model_name not in registry:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {list(registry)}"
            )

        # Retrieve this model's configuration.
        model_config = registry[model_name]

        # Prevent calls to models with placeholder identifiers.
        if model_config.model_id.startswith("REPLACE_WITH_"):
            print(f"Skipping {model_name}: model ID not configured")
            continue

        # Run the 10 questions for this model.
        model_results = await benchmark_model(
            model_name,
            model_config,
            PROMPTS
        )

        # Add the results to our complete benchmark.
        all_results.extend(model_results)

    # Create a reports directory if it does not exist.
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ----------------------------------------------------
    # REPLACE the existing output_file assignment below.
    # Keep it inside the main() function.
    # ----------------------------------------------------

    # Generate a unique timestamp for this benchmark run.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # Create a unique filename so previous results are preserved.
    output_file = reports_dir / f"benchmark_{timestamp}.json"


    # Save our benchmark results as formatted JSON.
    output_file.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8"
    )

    # Count the number of successful requests.
    successful = sum(
        result["status"] == "success"
        for result in all_results
    )

    # Calculate the total estimated API cost.
    total_cost = sum(
        result.get("cost_usd", 0)
        for result in all_results
    )

    # Display the benchmark summary.
    print("\n--- Benchmark Summary ---")
    print(f"Requests completed: {len(all_results)}")
    print(f"Successful requests: {successful}")
    print(f"Estimated cost (USD): ${total_cost:.6f}")
    print(f"Results saved to: {output_file}")


# Only execute the benchmark when this file is run directly.
if __name__ == "__main__":

    # Start the asynchronous benchmark.
    asyncio.run(main())
