
# Import JSON to read our labelled questions.
import json

# Import Path to locate our project files.
from pathlib import Path

# Import joblib to save our trained model.
import joblib

# Import DictVectorizer to convert feature dictionaries into numbers.
from sklearn.feature_extraction import DictVectorizer

# Import Logistic Regression, our machine-learning algorithm.
from sklearn.linear_model import LogisticRegression

# Import the tool for splitting training and testing data.
from sklearn.model_selection import train_test_split

# Import tools for measuring prediction performance.
from sklearn.metrics import accuracy_score, confusion_matrix

# Import our existing feature extractor.
from src.classifier.feature_extractor import extract_features


# Define the three valid complexity categories.
LABELS = ["simple", "medium", "complex"]


# Create the main training function.
def main():

    # Find our project's root folder.
    project_root = Path(__file__).resolve().parents[2]

    # Train using our 210-prompt V2 dataset.
    dataset_path = project_root / "data" / "complexity_dataset_v2.json"

    # Load our labelled examples from JSON.
    dataset = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    # Extract seven numerical features from each prompt.
    features = [
        extract_features(item["prompt"])
        for item in dataset
    ]

    # Collect the correct complexity labels.
    labels = [
        item["complexity"]
        for item in dataset
    ]

    # Split the data into 80% training and 20% testing.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )

    # Create a tool that converts dictionaries into feature columns.
    vectorizer = DictVectorizer(sparse=False)

    # Learn the feature columns using training data only.
    X_train_vectors = vectorizer.fit_transform(X_train)

    # Convert test data using those same columns.
    X_test_vectors = vectorizer.transform(X_test)

    # Create our Logistic Regression classifier.
    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    # Train the classifier using labelled examples.
    model.fit(X_train_vectors, y_train)

    # Predict the labels of unseen test questions.
    predictions = model.predict(X_test_vectors)

    # Calculate the percentage of correct predictions.
    accuracy = accuracy_score(y_test, predictions) * 100

    # Calculate how predictions are distributed by category.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    # Display the training results.
    print("\n--- Machine-Learning Classifier Evaluation ---")
    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Display the confusion matrix.
    print("\nConfusion Matrix")
    print("Labels:", LABELS)
    print(matrix)

    # Create a folder for saved machine-learning models.
    models_dir = project_root / "models"

    # Create the folder if it does not already exist.
    models_dir.mkdir(exist_ok=True)

    # Preserve the original model by saving the V2 version separately.
    model_path = models_dir / "complexity_classifier_v2.joblib"

    # Save both the model and its feature converter.
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "labels": LABELS,
        },
        model_path
    )

    # Display the location of our saved model.
    print(f"\nModel saved to: {model_path}")


# Run training when this file is executed directly.
if __name__ == "__main__":
    main()
