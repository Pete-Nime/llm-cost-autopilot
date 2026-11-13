
# Import JSON so we can read and save JSON files.
import json

# Import Path to work with file locations.
from pathlib import Path

# Import joblib to load our trained classifier.
import joblib

# Import pandas to create a table of prompt features.
import pandas as pd

# Import accuracy to measure correct predictions.
from sklearn.metrics import accuracy_score

# Import a report for precision, recall, and F1-score.
from sklearn.metrics import classification_report

# Import a confusion matrix to inspect prediction errors.
from sklearn.metrics import confusion_matrix

# Import the feature extractor we built earlier.
from src.classifier.feature_extractor import extract_features


# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate our benchmark questions.
BENCHMARK_PATH = (
    PROJECT_ROOT / "data" / "complexity_benchmark_v1.json"
)

# Locate the trained V2 TF-IDF classifier.
MODEL_PATH = (
    PROJECT_ROOT / "models" / "tfidf_complexity_classifier_v2.joblib"
)

# Decide where to save our evaluation report.
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "benchmark_evaluation_v1.json"
)

# Define the order of our complexity categories.
LABELS = ["simple", "medium", "complex"]


# Create a function to perform the evaluation.
def main():

    # Read the benchmark JSON file.
    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark_data = json.load(file)

    # Load the classifier we trained previously.
    model = joblib.load(MODEL_PATH)

    # Create an empty list to hold feature rows.
    feature_rows = []

    # Process every benchmark question.
    for item in benchmark_data:

        # Extract the question text.
        prompt = item["prompt"]

        # Calculate our seven numerical features.
        features = extract_features(prompt)

        # Include the original prompt for TF-IDF.
        row = {"prompt": prompt}

        # Add the numerical features to the same row.
        row.update(features)

        # Save the completed feature row.
        feature_rows.append(row)

    # Convert our feature rows into a pandas table.
    X = pd.DataFrame(feature_rows)

    # Collect the correct complexity labels.
    y_true = [
        item["complexity"] for item in benchmark_data
    ]

    # Ask the trained classifier to predict each label.
    y_pred = model.predict(X).tolist()

    # Request probability estimates for each prediction.
    probabilities = model.predict_proba(X)

    # Get the model's actual probability-column order.
    model_classes = model.classes_.tolist()

    # Calculate overall prediction accuracy.
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1-score.
    class_report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Calculate the confusion matrix.
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    # Create a list for incorrect predictions.
    mistakes = []

    # Count dangerous complex-to-simple errors.
    complex_to_simple = 0

    # Examine each prediction individually.
    for index, item in enumerate(benchmark_data):

        # Get the expected label.
        expected = y_true[index]

        # Get the predicted label.
        predicted = y_pred[index]

        # Check whether the prediction is incorrect.
        if expected != predicted:

            # Match probabilities to their class names.
            class_probabilities = {
                label: round(float(probabilities[index][position]), 4)
                for position, label in enumerate(model_classes)
            }

            # Store the details of this mistake.
            mistakes.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "expected": expected,
                "predicted": predicted,
                "probabilities": class_probabilities,
            })

            # Detect a complex question classified as simple.
            if expected == "complex" and predicted == "simple":
                complex_to_simple += 1

    # Create an empty list for every prediction.
    all_predictions = []

    # Visit every question in our benchmark.
    for index, item in enumerate(benchmark_data):

        # Match each probability to its correct class name.
        class_probabilities = {
            label: round(float(probabilities[index][position]), 4)
            for position, label in enumerate(model_classes)
        }

        # Store the question and the model's prediction.
        all_predictions.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "expected": y_true[index],
            "predicted": y_pred[index],
            "probabilities": class_probabilities,
        })

    # Assemble our final evaluation report.
    report = {
        "dataset": "complexity_benchmark_v1",
        "model": "tfidf_complexity_classifier_v2",
        "total_questions": len(benchmark_data),
        "accuracy": round(float(accuracy), 4),
        "labels": LABELS,
        "classification_report": class_report,
        "confusion_matrix": matrix.tolist(),
        "complex_to_simple_errors": complex_to_simple,
        "incorrect_predictions": mistakes,
        "all_predictions": all_predictions,
    }

    # Create the reports folder if it does not exist.
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save our report as readable JSON.
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Print a heading for the terminal results.
    print("\n--- V2 Benchmark Evaluation ---")

    # Print how many questions were evaluated.
    print("Questions:", len(benchmark_data))

    # Print the overall accuracy as a percentage.
    print(f"Accuracy: {accuracy:.2%}")

    # Print the number of incorrect predictions.
    print("Incorrect predictions:", len(mistakes))

    # Print the number of complex-to-simple mistakes.
    print("Complex classified as simple:", complex_to_simple)

    # Print the classification report for each category.
    print("\nClassification Report:")

    # Create a readable text version of the report.
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    # Print the order of the confusion matrix.
    print("Confusion matrix labels:", LABELS)

    # Print the confusion matrix itself.
    print(matrix)

    # Display each incorrect prediction.
    print("\nIncorrect Predictions:")

    # Loop through the recorded mistakes.
    for mistake in mistakes:

        # Show the question ID and the two labels.
        print(
            f"{mistake['id']}: "
            f"expected={mistake['expected']}, "
            f"predicted={mistake['predicted']}"
        )

    # Show where the report was saved.
    print("\nReport saved to:", REPORT_PATH)


# Run main only when this script is executed directly.
if __name__ == "__main__":
    main()
