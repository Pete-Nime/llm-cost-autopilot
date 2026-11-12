
# Import JSON so we can read and save structured data.
import json

# Import Path to find project files.
from pathlib import Path

# Import joblib to load our trained model.
import joblib

# Import pandas to organize prompt features.
import pandas as pd

# Import tools to measure classifier performance.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Import our existing numerical feature extractor.
from src.classifier.feature_extractor import extract_features


# Define the three complexity categories.
LABELS = ["simple", "medium", "complex"]


# Convert prompts into the format our V2 model expects.
def build_feature_table(prompts):

    # Create an empty list to hold our feature rows.
    rows = []

    # Process each prompt separately.
    for prompt in prompts:

        # Extract our seven numerical features.
        features = extract_features(prompt)

        # Include the original text for TF-IDF.
        features["prompt"] = prompt

        # Add the completed row to our list.
        rows.append(features)

    # Convert all rows into a pandas table.
    return pd.DataFrame(rows)


# Define our main evaluation function.
def main():

    # Find the root folder of LLM Cost Autopilot.
    project_root = Path(__file__).resolve().parents[2]

    # Locate our previously reviewed challenge questions.
    dataset_path = (
        project_root / "data" / "complexity_challenge.json"
    )

    # Locate the trained V2 TF-IDF classifier.
    model_path = (
        project_root
        / "models"
        / "tfidf_complexity_classifier_v2.joblib"
    )

    # Read the challenge dataset from disk.
    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    # Collect the prompt text.
    prompts = [
        item["prompt"]
        for item in dataset
    ]

    # Collect the expected complexity labels.
    expected_labels = [
        item["complexity"]
        for item in dataset
    ]

    # Load our trained classifier.
    model = joblib.load(model_path)

    # Convert prompts into text and numerical features.
    X_test = build_feature_table(prompts)

    # Ask the classifier to predict complexity.
    predictions = model.predict(X_test)

    # Ask the classifier for probabilities across all three classes.
    probabilities = model.predict_proba(X_test)

    # Retrieve the class order used by the trained model.
    model_classes = model.classes_


    # Calculate the overall accuracy.
    accuracy = accuracy_score(
        expected_labels,
        predictions,
    ) * 100

    # Calculate precision, recall and F1 for each label.
    metrics = classification_report(
        expected_labels,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Build a confusion matrix in our chosen label order.
    matrix = confusion_matrix(
        expected_labels,
        predictions,
        labels=LABELS,
    )

    # Identify complex prompts incorrectly labelled simple.
    dangerous_errors = []

    # Create a list containing every individual prediction.
    prediction_details = []

    # Examine each prompt and its predicted label.
    # Process each prompt, prediction and probability distribution together.
    for item, prediction, probability in zip(
        dataset,
        predictions,
        probabilities,
    ):

        # Create a dictionary to store one prediction.
        result = {

            # Save the unique question ID.
            "id": item["id"],

            # Save the original question.
            "prompt": item["prompt"],

            # Save the correct complexity category.
            "expected": item["complexity"],

            # Save the category predicted by our classifier.
            "predicted": str(prediction),

            # Check whether our prediction was correct.
            "correct": bool(
                item["complexity"] == prediction
            ),

            # Save the probability of the predicted category.
            "predicted_probability": round(
                float(max(probability)), 4
            ),

            # Save the probabilities for all three categories.
            "class_probabilities": {

                # Match each category name with its probability.
                str(label): round(float(score), 4)

                # Go through each category and its probability.
                for label, score in zip(
                    model_classes,
                    probability
                )
            },
        }


        # Save the individual result.
        prediction_details.append(result)

        # Check for our highest-risk classification error.
        if (
            item["complexity"] == "complex"
            and prediction == "simple"
        ):

            # Record the dangerous misclassification.
            dangerous_errors.append(result)

    # Build the complete evaluation report.
    report = {
        "model": model_path.name,
        "dataset": dataset_path.name,
        "evaluation_type": "previously_reviewed_challenge",
        "total_prompts": len(dataset),
        "accuracy_percent": round(accuracy, 2),
        "classification_metrics": metrics,
        "label_order": LABELS,
        "confusion_matrix": matrix.tolist(),
        "complex_to_simple_errors": dangerous_errors,
        "predictions": prediction_details,
    }

    # Locate our existing reports directory.
    reports_dir = project_root / "reports"

    # Create the directory if it does not exist.
    reports_dir.mkdir(exist_ok=True)

    # Choose a filename that identifies this evaluation.
    report_path = (
        reports_dir / "v2_challenge_evaluation.json"
    )

    # Save the report as readable JSON.
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # Display the evaluation heading.
    print("\n--- V2 Classifier Challenge Evaluation ---")

    # Display how many questions were evaluated.
    print(f"Challenge prompts: {len(dataset)}")

    # Display the overall classification accuracy.
    print(f"Accuracy: {accuracy:.2f}%")

    # Display the detailed classification metrics.
    print("\nClassification Report")

    # Format and display the metrics for readability.
    print(
        classification_report(
            expected_labels,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    # Display the confusion matrix.
    print("\nConfusion Matrix")

    # Explain the label order.
    print("Labels:", LABELS)

    # Display the matrix.
    print(matrix)

    # Display our highest-risk error count.
    print(
        "\nComplex classified as simple:",
        len(dangerous_errors),
    )

    # Display the heading for incorrect predictions.
    print("\nIncorrect Predictions")

    # Go through every prediction in our results.
    for result in prediction_details:

        # Only display predictions that were wrong.
        if not result["correct"]:

            # Show the question ID and expected/predicted labels.
            print(
                f"\n{result['id']}: "
                f"expected={result['expected']}, "
                f"predicted={result['predicted']}"
            )

            # Show the probability of the predicted class.
            print(
                f"Predicted class probability: "
                f"{result['predicted_probability'] * 100:.2f}%"
            )

            # Display the probability breakdown heading.
            print("All class probabilities:")

            # Loop through the three complexity categories.
            for label, probability in result[
                "class_probabilities"
            ].items():

                # Display the probability as a percentage.
                print(
                    f"  {label}: {probability * 100:.2f}%"
                )

    # Confirm the saved report location.
    print(f"\nReport saved to: {report_path}")


# Run the evaluation when this script is executed.
if __name__ == "__main__":
    main()
