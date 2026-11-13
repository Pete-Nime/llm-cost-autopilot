
# Import pytest to run automated tests.
import pytest

# Import the policy from its permanent location.
from src.router.risk_policy import apply_policy


# Test that a medium prediction escalates at the threshold.
def test_medium_escalates_to_complex():

    # Create a medium prediction with high complex probability.
    prediction = {
        "predicted": "medium",
        "probabilities": {"complex": 0.45},
    }

    # Apply the policy using a 40% threshold.
    result = apply_policy(prediction, 0.40)

    # Confirm that the route becomes complex.
    assert result == "complex"


# Test that a medium prediction stays medium below threshold.
def test_medium_stays_medium():

    # Create a medium prediction with low complex probability.
    prediction = {
        "predicted": "medium",
        "probabilities": {"complex": 0.25},
    }

    # Apply the policy using a 40% threshold.
    result = apply_policy(prediction, 0.40)

    # Confirm that the route stays medium.
    assert result == "medium"


# Test that simple predictions remain simple.
def test_simple_prediction_unchanged():

    # Create a simple prediction.
    prediction = {
        "predicted": "simple",
        "probabilities": {"complex": 0.45},
    }

    # Apply the policy.
    result = apply_policy(prediction, 0.40)

    # Confirm that this policy does not change simple predictions.
    assert result == "simple"


# Test that complex predictions remain complex.
def test_complex_prediction_unchanged():

    # Create a complex prediction.
    prediction = {
        "predicted": "complex",
        "probabilities": {"complex": 0.75},
    }

    # Apply the policy.
    result = apply_policy(prediction, 0.40)

    # Confirm that complex remains complex.
    assert result == "complex"


# Test the exact threshold boundary.
def test_probability_equal_to_threshold():

    # Create a medium prediction at exactly 40%.
    prediction = {
        "predicted": "medium",
        "probabilities": {"complex": 0.40},
    }

    # Apply the policy.
    result = apply_policy(prediction, 0.40)

    # Confirm that equality triggers escalation.
    assert result == "complex"


# Test multiple threshold values.
@pytest.mark.parametrize(
    "threshold,expected",
    [
        (0.20, "complex"),
        (0.30, "complex"),
        (0.40, "medium"),
        (0.50, "medium"),
    ],
)
def test_different_thresholds(threshold, expected):

    # Create a prediction with 35% complex probability.
    prediction = {
        "predicted": "medium",
        "probabilities": {"complex": 0.35},
    }

    # Apply the selected threshold.
    result = apply_policy(prediction, threshold)

    # Confirm that the expected route is returned.
    assert result == expected
