
# Import JSON to read and save our evaluation data.
import json

# Import Path to locate files in our project.
from pathlib import Path

# Import our reusable routing safety policy.
from src.router.risk_policy import apply_policy

# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate the report containing all 90 predictions.
INPUT_PATH = (
    PROJECT_ROOT / "reports" / "benchmark_evaluation_v1.json"
)

# Decide where our routing analysis will be saved.
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "routing_policy_analysis.json"
)

# Define the complexity categories in order.
LABELS = ["simple", "medium", "complex"]

# Define candidate thresholds to investigate.
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# Create a function to evaluate one threshold.
def evaluate_threshold(predictions, threshold):

    # Count all questions routed to complex.
    routed_to_complex = 0

    # Count predictions changed by our safety rule.
    escalations = 0

    # Count actual complex questions routed too low.
    complex_underestimations = 0

    # Count simple or medium questions routed as complex.
    unnecessary_complex_routes = 0

    # Count correct routing decisions.
    correct_routes = 0

    # Examine each saved benchmark prediction.
    for prediction in predictions:

        # Get the correct label from the benchmark.
        expected = prediction["expected"]

        # Get the classifier's original prediction.
        original = prediction["predicted"]

        # Apply our candidate routing policy.
        routed = apply_policy(prediction, threshold)

        # Check whether the safety rule changed the route.
        if routed != original:
            escalations += 1

        # Count all questions sent to the complex route.
        if routed == "complex":
            routed_to_complex += 1

        # Count actual complex tasks routed below complex.
        if expected == "complex" and routed != "complex":
            complex_underestimations += 1

        # Count non-complex tasks sent to the complex route.
        if expected != "complex" and routed == "complex":
            unnecessary_complex_routes += 1

        # Check whether the final route matches the label.
        if routed == expected:
            correct_routes += 1

    # Count the total number of benchmark questions.
    total = len(predictions)

    # Return all metrics for this threshold.
    return {
        "threshold": threshold,
        "accuracy": round(correct_routes / total, 4),
        "escalations": escalations,
        "routed_to_complex": routed_to_complex,
        "complex_underestimations": complex_underestimations,
        "unnecessary_complex_routes": unnecessary_complex_routes,
    }


# Create the main function.
def main():

    # Open our existing benchmark evaluation report.
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    # Retrieve all 90 predictions.
    predictions = report["all_predictions"]

    # Confirm that the report contains predictions.
    if not predictions:
        raise ValueError("No predictions found in the report.")

    # Check that each record contains required fields.
    for prediction in predictions:

        # Verify the expected and predicted labels.
        if (
            prediction["expected"] not in LABELS
            or prediction["predicted"] not in LABELS
        ):
            raise ValueError("Unknown complexity label found.")

        # Verify the complex probability exists.
        probability = prediction["probabilities"]["complex"]

        # Reject probabilities outside the valid range.
        if not 0 <= probability <= 1:
            raise ValueError("Invalid complex probability.")

    # Calculate the baseline with no safety rule.
    baseline_underestimations = sum(
        item["expected"] == "complex"
        and item["predicted"] != "complex"
        for item in predictions
    )

    # Count baseline non-complex tasks sent to complex.
    baseline_unnecessary = sum(
        item["expected"] != "complex"
        and item["predicted"] == "complex"
        for item in predictions
    )

    # Create an empty list for threshold results.
    results = []

    # Test every candidate threshold.
    for threshold in THRESHOLDS:

        # Evaluate the policy using this threshold.
        result = evaluate_threshold(
            predictions,
            threshold,
        )

        # Save the result for comparison.
        results.append(result)

    # Build the complete analysis report.
    output = {
        "dataset": "complexity_benchmark_v1",
        "total_questions": len(predictions),
        "baseline_accuracy": report["accuracy"],
        "baseline_complex_underestimations": baseline_underestimations,
        "baseline_unnecessary_complex_routes": baseline_unnecessary,
        "policy": "Escalate medium to complex when P(complex) >= threshold",
        "threshold_results": results,
    }

    # Create the reports directory if needed.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the results to a JSON file.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    # Print the analysis heading.
    print("\n--- Risk-Aware Routing Analysis ---")

    # Print the number of benchmark questions.
    print("Questions:", len(predictions))

    # Print the original classifier accuracy.
    print(f"Baseline accuracy: {report['accuracy']:.2%}")

    # Print the original complex underestimations.
    print(
        "Baseline complex underestimations:",
        baseline_underestimations,
    )

    # Print the original unnecessary complex routes.
    print(
        "Baseline unnecessary complex routes:",
        baseline_unnecessary,
    )

    # Print a heading for the threshold comparison.
    print("\nThreshold | Accuracy | Escalations | Complex missed | Unnecessary complex")

    # Display the result for each threshold.
    for result in results:

        # Format the threshold as a percentage.
        threshold_text = f"{result['threshold']:.0%}"

        # Format the resulting routing accuracy.
        accuracy_text = f"{result['accuracy']:.2%}"

        # Print the metrics in one row.
        print(
            f"{threshold_text:>9} | "
            f"{accuracy_text:>8} | "
            f"{result['escalations']:>11} | "
            f"{result['complex_underestimations']:>14} | "
            f"{result['unnecessary_complex_routes']:>19}"
        )

    # Print the saved report location.
    print("\nReport saved to:", OUTPUT_PATH)


# Run main when this file is executed as a script.
if __name__ == "__main__":
    main()
