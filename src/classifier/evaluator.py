
# Import JSON to read our labelled questions.
import json

# Import Path to locate project files.
from pathlib import Path

# Import our existing complexity classifier.
from src.classifier.complexity_classifier import classify_complexity


# Define the three possible complexity labels.
LABELS = ("simple", "medium", "complex")


# Create a function that evaluates our classifier.
def evaluate_classifier(dataset_path):

    # Convert the supplied file location into a Path.
    dataset_path = Path(dataset_path)

    # Load the dataset from its JSON file.
    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    # Reject an empty dataset.
    if not dataset:
        raise ValueError("Dataset cannot be empty.")

    # Create counters for correct and total predictions.
    correct = 0
    total = 0

    # Create a list to store incorrectly classified questions.
    mistakes = []

    # Create an empty confusion matrix.
    # Rows represent expected labels.
    # Columns represent predicted labels.
    confusion_matrix = {
        expected: {predicted: 0 for predicted in LABELS}
        for expected in LABELS
    }

    # Examine each question in the dataset.
    for item in dataset:

        # Read the question.
        prompt = item["prompt"]

        # Read the expected complexity label.
        expected = item["complexity"]

        # Reject unknown reference labels.
        if expected not in LABELS:
            raise ValueError(f"Unknown label: {expected}")

        # Ask our classifier to predict the difficulty.
        predicted = classify_complexity(prompt)

        # Count this question.
        total += 1

        # Record the prediction in the confusion matrix.
        confusion_matrix[expected][predicted] += 1

        # Check whether the prediction was correct.
        if predicted == expected:

            # Increase the number of correct predictions.
            correct += 1

        else:

            # Save details about this mistake.
            mistakes.append({
                "id": item["id"],
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted
            })

    # Calculate accuracy as a percentage.
    accuracy = (correct / total) * 100

    # Return the complete evaluation results.
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "mistakes": mistakes
    }


# Define the function that runs when we execute this file.
def main():

    # Find the root folder of our existing project.
    project_root = Path(__file__).resolve().parents[2]

    # Locate our labelled question dataset.
    dataset_path = project_root / "data" / "complexity_dataset.json"

    # Evaluate the classifier using all questions.
    results = evaluate_classifier(dataset_path)

    # Display the evaluation summary.
    print("\n--- Complexity Classifier Evaluation ---")
    print(f"Total questions: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")

    # Display the confusion matrix.
    print("\nConfusion Matrix (Expected -> Predicted)")

    # Print each expected category and its predictions.
    for expected, predictions in results["confusion_matrix"].items():
        print(f"{expected}: {predictions}")

    # Display how many questions were classified incorrectly.
    print(f"\nMisclassified questions: {len(results['mistakes'])}")

    # Show up to five mistakes for inspection.
    for mistake in results["mistakes"][:5]:
        print(
            f"{mistake['id']}: "
            f"Expected={mistake['expected']}, "
            f"Predicted={mistake['predicted']}"
        )
        print(f"  Prompt: {mistake['prompt']}")


# Run main only when this file is executed as a program.
if __name__ == "__main__":
    main()
