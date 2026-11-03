
# Import dataclass to create structured verification results.
from dataclasses import dataclass


# Define the structure of our quality verification result.
@dataclass(frozen=True)
class QualityResult:

    # True if the answer meets our basic scoring threshold.
    passed: bool

    # Explain why the answer passed or failed.
    reason: str

    # Store a basic length-based score between 0.0 and 1.0.
    score: float = 0.0


# Check an AI-generated answer using a simple length-based rule.
def verify_quality(
    answer: str,
    minimum_score: float = 0.5,
) -> QualityResult:

    # Ensure the threshold is numeric.
    if not isinstance(minimum_score, (int, float)):
        raise TypeError("Minimum score must be numeric.")

    # Ensure the threshold is between 0.0 and 1.0.
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("Minimum score must be between 0 and 1.")

    # Reject answers that are not strings.
    if not isinstance(answer, str):
        return QualityResult(
            passed=False,
            reason="AI response must be a string.",
            score=0.0,
        )

    # Remove spaces from the beginning and end.
    cleaned_answer = answer.strip()

    # Reject empty answers.
    if not cleaned_answer:
        return QualityResult(
            passed=False,
            reason="AI response is empty.",
            score=0.0,
        )

    # Count the words in the answer.
    word_count = len(cleaned_answer.split())

    # Calculate a score between 0.0 and 1.0.
    # Answers with 20 or more words receive 1.0.
    score = min(word_count / 20, 1.0)

    # Check whether the score meets the required threshold.
    passed = score >= minimum_score

    # Explain how the score was calculated.
    reason = (
        f"Answer contains {word_count} words. "
        f"Completeness score: {score:.2f}. "
        f"Required threshold: {minimum_score:.2f}."
    )

    # Return the final verification result.
    return QualityResult(
        passed=passed,
        reason=reason,
        score=score,
    )


# Provide an asynchronous interface for quality verification.
async def verify_quality_async(
    answer: str,
    minimum_score: float = 0.5,
) -> QualityResult:

    # Reuse our existing synchronous verification function.
    result = verify_quality(
        answer=answer,
        minimum_score=minimum_score,
    )

    # Return the verification result to the caller.
    return result
