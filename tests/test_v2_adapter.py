
# Import pytest to check expected errors.
import pytest

# Import the V2 classifier adapter.
from src.classifier.v2_adapter import (
    load_v2_model,
    predict_v2_complexity,
)


# Verify that the trained V2 model loads.
def test_v2_model_loads():

    # Load our saved classifier.
    model = load_v2_model()

    # Confirm that the model can make predictions.
    assert hasattr(model, "predict")

    # Confirm that the model returns probabilities.
    assert hasattr(model, "predict_proba")


# Verify that the classifier returns a valid category.
def test_v2_prediction_has_valid_category():

    # Classify a straightforward question.
    result = predict_v2_complexity(
        "What is the capital of Japan?"
    )

    # Confirm that the result uses a known category.
    assert result.complexity in {
        "simple",
        "medium",
        "complex",
    }


# Verify that all three probabilities are returned.
def test_v2_prediction_has_probabilities():

    # Classify a sample question.
    result = predict_v2_complexity(
        "Explain how database indexes affect query performance."
    )

    # Confirm that every category has a probability.
    assert set(result.probabilities) == {
        "simple",
        "medium",
        "complex",
    }

    # Confirm that the probabilities add up to one.
    assert sum(result.probabilities.values()) == pytest.approx(
        1.0
    )


# Verify that empty prompts are rejected.
def test_v2_rejects_empty_prompt():

    # Expect our shared validator to reject empty text.
    with pytest.raises(ValueError):

        # Attempt to classify an empty prompt.
        predict_v2_complexity("")


# Verify that the cached model is reused.
def test_v2_model_is_cached():

    # Load the model for the first time.
    first_model = load_v2_model()

    # Request the model again.
    second_model = load_v2_model()

    # Confirm both references point to the same object.
    assert first_model is second_model
