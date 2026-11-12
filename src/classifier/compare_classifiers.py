
# Import JSON to read our labelled dataset.
import json

# Import Path to locate our project files.
from pathlib import Path

# Import joblib to load our saved ML model.
import joblib

# Import the same data splitting function used in training.
from sklearn.model_selection import train_test_split

# Import accuracy calculation.
from sklearn.metrics import accuracy_score

# Import our original rule-based classifier.
from src.classifier.complexity_classifier import classify_complexity

# Import our existing feature extractor.
from src.classifier.feature_extractor import extract_features


# Create our main comparison function.
def main():

    # Find the root folder of our project.
    project_root = Path(__file__).resolve().parents[2]

    # Locate the existing dataset.
    dataset_path = project_root / "data" / "complexity_dataset.json"

    # Read all 210 labelled prompts.
    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    # Collect the correct complexity labels.
    labels = [
        item["complexity"] for item in dataset
    ]

    # Recreate the exact training/test split.
    _, test_items, _, test_labels = train_test_split(
        dataset,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    # Load our previously trained model and vectorizer.
    saved = joblib.load(
        project_root / "models" / "complexity_classifier.joblib"
    )

    # Retrieve the trained classifier.
    model = saved["model"]

    # Retrieve the fitted feature vectorizer.
    vectorizer = saved["vectorizer"]

    # Extract features from the held-out prompts.
    test_features = [
        extract_features(item["prompt"])
        for item in test_items
    ]

    # Convert features into the columns learned during training.
    test_vectors = vectorizer.transform(test_features)

    # Ask the trained ML model for predictions.
    ml_predictions = model.predict(test_vectors)

    # Ask the original rule-based classifier for predictions.
    rule_predictions = [
        classify_complexity(item["prompt"])
        for item in test_items
    ]

    # Calculate ML classifier accuracy.
    ml_accuracy = accuracy_score(
        test_labels, ml_predictions
    ) * 100

    # Calculate rule-based classifier accuracy.
    rule_accuracy = accuracy_score(
        test_labels, rule_predictions
    ) * 100

    # Display the results.
    print("\n--- Classifier Comparison ---")
    print(f"Test prompts: {len(test_items)}")
    print(f"Rule-based accuracy: {rule_accuracy:.2f}%")
    print(f"Machine-learning accuracy: {ml_accuracy:.2f}%")

    # Display which classifier performed better.
    if ml_accuracy > rule_accuracy:
        print("Winner: Machine-learning classifier")
    elif rule_accuracy > ml_accuracy:
        print("Winner: Rule-based classifier")
    else:
        print("Result: Both classifiers achieved equal accuracy")


# Run the comparison when executed directly.
if __name__ == "__main__":
    main()
