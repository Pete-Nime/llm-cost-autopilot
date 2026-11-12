
# Import JSON to read our labelled prompts.
import json

# Import Path to locate our project files.
from pathlib import Path

# Import joblib to save our trained model.
import joblib

# Import pandas to organize text and numerical features.
import pandas as pd

# Import the tool that combines different feature types.
from sklearn.compose import ColumnTransformer

# Import TF-IDF to convert words into numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Logistic Regression to learn classification patterns.
from sklearn.linear_model import LogisticRegression

# Import tools to measure classification performance.
from sklearn.metrics import accuracy_score, confusion_matrix

# Import the tool that splits our dataset.
from sklearn.model_selection import train_test_split

# Import Pipeline to connect our processing steps.
from sklearn.pipeline import Pipeline

# Import StandardScaler to scale numerical features.
from sklearn.preprocessing import StandardScaler

# Import our existing feature extractor.
from src.classifier.feature_extractor import extract_features


# Define the three complexity labels in a fixed order.
LABELS = ["simple", "medium", "complex"]

# Define the numerical features used by our original model.
FEATURE_NAMES = [
    "word_count",
    "character_count",
    "sentence_count",
    "instruction_count",
    "constraint_count",
    "output_format_count",
    "context_length",
]


# Convert a collection of prompts into a feature table.
def build_feature_table(prompts):

    # Create an empty collection for the table rows.
    rows = []

    # Process each prompt individually.
    for prompt in prompts:

        # Extract the seven numerical features.
        features = extract_features(prompt)

        # Add the original text for TF-IDF.
        features["prompt"] = prompt

        # Store the completed feature row.
        rows.append(features)

    # Convert our rows into a pandas DataFrame.
    return pd.DataFrame(rows)


# Define the main training function.
def main():

    # Find the root directory of our project.
    project_root = Path(__file__).resolve().parents[2]

    # Load our new 210-prompt V2 dataset instead of the original dataset.
    dataset_path = project_root / "data" / "complexity_dataset_v2.json"

    # Read the labelled dataset.
    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    # Collect the original questions.
    prompts = [
        item["prompt"]
        for item in dataset
    ]

    # Collect the correct complexity labels.
    labels = [
        item["complexity"]
        for item in dataset
    ]

    # Recreate the same 80/20 split as our original model.
    train_prompts, test_prompts, y_train, y_test = train_test_split(
        prompts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    # Extract features from the training questions.
    X_train = build_feature_table(train_prompts)

    # Extract features from the held-out test questions.
    X_test = build_feature_table(test_prompts)

    # Configure our text and numerical feature processors.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                ),
                "prompt",
            ),
            (
                "numeric",
                StandardScaler(),
                FEATURE_NAMES,
            ),
        ]
    )

    # Connect feature processing to the ML classifier.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Learn from training examples only.
    pipeline.fit(X_train, y_train)

    # Predict the labels of the held-out questions.
    predictions = pipeline.predict(X_test)

    # Calculate the accuracy percentage.
    accuracy = accuracy_score(y_test, predictions) * 100

    # Calculate the confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    # Display the results.
    print("\n--- TF-IDF Classifier Evaluation ---")
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Display the confusion matrix.
    print("\nConfusion Matrix")
    print("Labels:", LABELS)
    print(matrix)

    # Create the models directory if necessary.
    models_dir = project_root / "models"

    # Ensure the directory exists.
    models_dir.mkdir(exist_ok=True)

    # Save V2 separately so the original trained model is not overwritten.
    model_path = models_dir / "tfidf_complexity_classifier_v2.joblib"

    # Save the entire fitted pipeline.
    joblib.dump(pipeline, model_path)

    # Confirm where the model was saved.
    print(f"\nTF-IDF model saved to: {model_path}")


# Run the training function when executed directly.
if __name__ == "__main__":
    main()
