
# Import JSON to read our saved benchmark reports.
import json

# Import Path to work with folders and files.
from pathlib import Path

# Find our project folder from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate the existing benchmark reports folder.
REPORTS_DIR = PROJECT_ROOT / "reports"

# Define the five models we want to compare.
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "llama3.2",
]


# Create a function to load benchmark reports.
def load_reports():

    # Store the newest report for each model.
    latest_reports = {}

    # Find every benchmark JSON file.
    files = REPORTS_DIR.glob("benchmark_*.json")

    # Examine every report.
    for file in files:

        # Read the JSON data.
        data = json.loads(
            file.read_text(encoding="utf-8")
        )

        # Skip files that do not contain request lists.
        if not isinstance(data, list) or not data:
            continue

        # Read the model name from the first request.
        model = data[0].get(
            "model_id",
            data[0].get("model")
        )

        # Ignore models outside our five-model registry.
        if model not in MODELS:
            continue

        # Get the modification time of this report.
        modified = file.stat().st_mtime

        # Check whether this model already has a report.
        previous = latest_reports.get(model)

        # Keep the most recently modified report.
        if previous is None or modified > previous[0]:
            latest_reports[model] = (modified, file, data)

    # Return our selected reports.
    return latest_reports


# Create a function to calculate benchmark statistics.
def calculate_statistics(data):

    # Keep only successful requests.
    successful = [
        item for item in data
        if item.get("status") == "success"
    ]

    # Count successful requests.
    count = len(successful)

    # Return safe values when no requests succeeded.
    if count == 0:
        return 0, 0.0, 0.0

    # Calculate the average response time.
    average_latency = sum(
        item["latency_ms"] for item in successful
    ) / count

    # Calculate total estimated API cost.
    total_cost = sum(
        item["cost_usd"] for item in successful
    )

    # Return success count, latency and cost.
    return count, average_latency, total_cost


# Create our main comparison function.
def main():

    # Load the latest available benchmark reports.
    reports = load_reports()

    # Print a heading for the comparison.
    print("\nLLM COST AUTOPILOT - PHASE 1 BASELINE")
    print("-" * 76)

    # Print the table headings.
    print(
        f"{'Model':<24}"
        f"{'Success':<12}"
        f"{'Avg Latency':<18}"
        f"{'Est. Cost (USD)':<18}"
    )

    # Print a separator below the headings.
    print("-" * 76)

    # Display each model in our registry.
    for model in MODELS:

        # Get the latest report for this model.
        report = reports.get(model)

        # Show a message if no report exists.
        if report is None:
            print(f"{model:<24}No benchmark found")
            continue

        # Extract the report's request data.
        data = report[2]

        # Calculate the report's statistics.
        count, latency, cost = calculate_statistics(data)

        # Count all requests, including failures.
        total = len(data)

        # Print the model's benchmark results.
        print(
            f"{model:<24}"
            f"{f'{count}/{total}':<12}"
            f"{f'{latency:.2f} ms':<18}"
            f"${cost:.6f}"
        )

    # Print the final separator.
    print("-" * 76)


# Run our comparison when this file is executed.
if __name__ == "__main__":
    main()
