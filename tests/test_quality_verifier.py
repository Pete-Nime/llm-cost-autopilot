
# Import our quality verification function.
from src.verifier.quality_verifier import verify_quality

# Import asyncio to run asynchronous code in our test.
import asyncio

# Import our new asynchronous verifier.
from src.verifier.quality_verifier import verify_quality_async


# Check that a valid answer passes a suitable threshold.
def test_valid_answer():

    # Create a short but valid example answer.
    answer = "Python is a programming language."

    # Verify the answer using a threshold suitable for this test.
    result = verify_quality(
        answer,
        minimum_score=0.2,
    )

    # Confirm the answer passes the chosen threshold.
    assert result.passed is True

    # Confirm the calculated score is correct.
    assert result.score == 0.25



# Check that an empty answer fails.
def test_empty_answer():

    # Send an empty answer.
    result = verify_quality("")

    # Confirm the answer was rejected.
    assert result.passed is False


# Check that whitespace-only answers fail.
def test_whitespace_answer():

    # Send an answer containing only spaces.
    result = verify_quality("   ")

    # Confirm the answer was rejected.
    assert result.passed is False


# Check that non-text responses fail.
def test_invalid_answer_type():

    # Send a number instead of text.
    result = verify_quality(123)

    # Confirm the invalid answer was rejected.
    assert result.passed is False


# Verify that a sufficiently long answer meets the threshold.
def test_long_answer_passes_threshold():

    # Create a sample answer with more than 20 words.
    answer = (
        "Python is a programming language used for automation, "
        "data analysis, artificial intelligence, web development, "
        "scientific computing, and many other software engineering tasks."
    )

    # Verify the answer.
    result = verify_quality(answer)

    # Confirm the answer meets the threshold.
    assert result.passed is True

    # Confirm the maximum completeness score.
    assert result.score == 1.0


# Confirm that asynchronous verification returns the expected result.
def test_async_quality_verification():

    # Create a sample answer containing enough words.
    answer = (
        "Python is a popular programming language used for "
        "building applications, automating tasks, processing "
        "data, developing machine learning models, and more."
    )

    # Execute the asynchronous verifier.
    result = asyncio.run(
        verify_quality_async(
            answer=answer,
            minimum_score=0.5,
        )
    )

    # Confirm that the answer passes the threshold.
    assert result.passed is True

    # Confirm that the result contains a numeric score.
    assert isinstance(result.score, float)


# Verify that a short answer fails a higher threshold.
def test_short_answer_fails_high_threshold():

    # Verify a short answer using a strict threshold.
    result = verify_quality(
        "Python is useful.",
        minimum_score=0.8,
    )

    # Confirm the answer does not meet the threshold.
    assert result.passed is False

    # Confirm the score is below the threshold.
    assert result.score < 0.8
