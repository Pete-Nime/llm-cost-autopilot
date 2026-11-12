
# Import joblib to load our trained model.
import joblib

# Import pandas to build the model's input table.
import pandas as pd

# Import Path to locate project files.
from pathlib import Path

# Import our existing numerical feature extractor.
from src.classifier.feature_extractor import extract_features


# Find our project root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate the saved V2 TF-IDF model.
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tfidf_complexity_classifier_v2.joblib"
)


# Define five new diagnostic prompts.
TEST_PROMPTS = [
    "What is the capital of Japan?",

    "Please read this message carefully and return only the invoice number. The customer explains that the payment was processed yesterday, the delivery address was updated, and the invoice reference is INV-90321.",

    "Compare two subscription plans based on monthly price, annual cost, included features and suitability for a small business.",

    "Write Python code to group sales transactions by month, calculate total revenue and identify the highest-performing month.",

    "Investigate why a distributed payment processing system produces duplicate transactions during regional failover and propose a reliable recovery strategy."
]


# Convert prompts into the same format used during training.
def build_feature_table(prompts):

    # Create an empty list for the feature rows.
    rows = []

    # Process every prompt.
    for prompt in prompts:

        # Extract the seven numerical features.
        features = extract_features(prompt)

        # Add the original prompt for TF-IDF.
        features["prompt"] = prompt

        # Store the completed row.
        rows.append(features)

    # Return the feature table.
    return pd.DataFrame(rows)


# Define our main testing function.
def main():

    # Load our saved V2 classifier.
    model = joblib.load(MODEL_PATH)

    # Convert the new prompts into model inputs.
    X_test = build_feature_table(TEST_PROMPTS)

    # Ask the model to predict complexity labels.
    predictions = model.predict(X_test)

    # Calculate prediction probabilities for each label.
    probabilities = model.predict_proba(X_test)

    # Display the report heading.
    print("\n--- V2 Diagnostic Prediction Report ---")

    # Process each prompt and its prediction.
    for index, (prompt, prediction) in enumerate(
        zip(TEST_PROMPTS, predictions)
    ):

        # Find the highest predicted class probability.
        confidence = max(probabilities[index]) * 100

        # Display the prompt number.
        print(f"\nPrompt {index + 1}")

        # Display the original question.
        print(f"Question: {prompt}")

        # Display the predicted category.
        print(f"Prediction: {prediction}")

        # Display the highest class probability.
        print(f"Model probability: {confidence:.2f}%")


# Run the script when executed directly.
if __name__ == "__main__":
    main()
