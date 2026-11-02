
# Import pytest for testing exceptions.
import pytest

# Import the classifier we just created.
from src.classifier.complexity_classifier import classify_complexity


# Check that a basic question is classified as simple.
def test_simple_question():

    # Give the classifier a straightforward definition question.
    result = classify_complexity("What is Python?")

    # Verify the expected difficulty.
    assert result == "simple"


# Check that an explanation request is medium.
def test_medium_question():

    # Ask for an explanation of an ETL pipeline.
    result = classify_complexity("Explain how an ETL pipeline works.")

    # Verify the expected difficulty.
    assert result == "medium"


# Check that a system design question is complex.
def test_complex_question():

    # Ask for a fault-tolerant system design.
    result = classify_complexity("Design a fault-tolerant system.")

    # Verify the expected difficulty.
    assert result == "complex"


# Check that empty questions are rejected.
def test_empty_question():

    # Expect a ValueError for an empty prompt.
    with pytest.raises(ValueError):

        # Send an empty string to the classifier.
        classify_complexity("")


# Check that non-string inputs are rejected.
def test_invalid_input():

    # Expect a TypeError for a number.
    with pytest.raises(TypeError):

        # Send an invalid input type.
        classify_complexity(123)


# Test natural-language questions that previously caused mistakes.
@pytest.mark.parametrize(
    "prompt, expected",
    [
        # A straightforward definition question.
        ("What does RAM stand for?", "simple"),

        # A question requiring an explanation of SQL behaviour.
        (
            "Why does a SQL JOIN sometimes produce duplicate rows?",
            "medium"
        ),

        # A production incident requiring investigation.
        (
            "My production database crashes during peak traffic. "
            "How can I diagnose and prevent recurring outages?",
            "complex"
        ),

        # An architecture decision involving several constraints.
        (
            "Evaluate whether we should migrate our data platform "
            "from PostgreSQL to Snowflake, considering performance, "
            "cost, security, and maintenance.",
            "complex"
        ),

        # An explanation of a multi-stage data pipeline.
        (
            "Walk me through the stages of an ETL pipeline "
            "and explain why each stage matters.",
            "medium"
        ),
    ]
)
def test_natural_language_complexity(prompt, expected):

    # Ask our classifier to predict the question difficulty.
    actual = classify_complexity(prompt)

    # Confirm the prediction matches the expected difficulty.
    assert actual == expected
