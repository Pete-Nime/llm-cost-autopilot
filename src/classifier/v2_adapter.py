
# Import dataclass to organize our prediction results.
from dataclasses import dataclass

# Import lru_cache to load the model only once.
from functools import lru_cache

# Import Path to locate our trained model.
from pathlib import Path

# Import joblib to load our saved machine-learning model.
import joblib

# Import pandas to prepare our model input.
import pandas as pd

# Import the feature extractor used during training.
from src.classifier.feature_extractor import extract_features

# Import our shared prompt validation function.
from src.router.prompt_validator import validate_prompt


# Find the main LLM Cost Autopilot project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate our trained V2 classifier file.
MODEL_PATH = (
    PROJECT_ROOT / "models" / "tfidf_complexity_classifier_v2.joblib"
)


# Define a structure to hold classification results.
@dataclass(frozen=True)
class V2Prediction:

    # Store the predicted complexity category.
    complexity: str

    # Store the estimated probability of each category.
    probabilities: dict[str, float]


# Cache the trained model after loading it.
@lru_cache(maxsize=1)
def load_v2_model():

    # Check that the saved model file exists.
    if not MODEL_PATH.is_file():

        # Stop with a helpful error if the model is missing.
        raise FileNotFoundError(
            f"V2 classifier model not found: {MODEL_PATH}"
        )

    # Load the trained machine-learning pipeline.
    return joblib.load(MODEL_PATH)


# Create a function to classify one incoming prompt.
def predict_v2_complexity(prompt: str) -> V2Prediction:

    # Validate and clean the incoming prompt.
    clean_prompt = validate_prompt(prompt)

    # Extract the numerical features used during training.
    features = extract_features(clean_prompt)

    # Combine the original prompt with its features.
    input_row = {
        "prompt": clean_prompt,
        **features,
    }

    # Convert the data into a one-row pandas DataFrame.
    input_data = pd.DataFrame([input_row])

    # Load our trained V2 classifier.
    model = load_v2_model()

    # Predict the most likely complexity category.
    predicted_label = str(model.predict(input_data)[0])

    # Calculate the probabilities of each category.
    probability_values = model.predict_proba(input_data)[0]

    # Match each category name with its probability.
    probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            model.classes_,
            probability_values,
        )
    }

    # Return the predicted category and its probabilities.
    return V2Prediction(
        complexity=predicted_label,
        probabilities=probabilities,
    )
