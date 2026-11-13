
# Define a reusable routing safety rule.
def apply_policy(prediction, threshold: float = 0.40) -> str:

    # Check that the threshold is a valid probability.
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    # Read the original complexity prediction.
    predicted = prediction["predicted"]

    # Read the model's estimated complex probability.
    complex_probability = prediction["probabilities"]["complex"]

    # Check whether the prediction is medium and the
    # complex probability reaches our escalation threshold.
    if (
        predicted == "medium"
        and complex_probability >= threshold
    ):

        # Escalate the routing decision to complex.
        return "complex"

    # Otherwise keep the original complexity.
    return predicted
