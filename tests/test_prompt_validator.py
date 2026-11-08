
# Import pytest for checking expected exceptions.
import pytest

# Import our shared prompt-length setting.
from config.budget import MAX_PROMPT_CHARACTERS

# Import the validator we want to test.
from src.router.prompt_validator import validate_prompt


# Check that valid prompts are returned without changes.
def test_valid_prompt():

    # Validate a normal question.
    result = validate_prompt("What is Python?")

    # Confirm the question is unchanged.
    assert result == "What is Python?"


# Check that surrounding whitespace is removed.
def test_prompt_is_trimmed():

    # Validate a question with extra spaces.
    result = validate_prompt("  What is Python?  ")

    # Confirm the extra spaces were removed.
    assert result == "What is Python?"


# Check that empty prompts are rejected.
def test_empty_prompt():

    # Expect a ValueError for an empty prompt.
    with pytest.raises(ValueError):
        validate_prompt("")


# Check that whitespace-only prompts are rejected.
def test_whitespace_prompt():

    # Expect a ValueError when no meaningful text exists.
    with pytest.raises(ValueError):
        validate_prompt("   ")


# Check that non-string inputs are rejected.
def test_invalid_prompt_type():

    # Expect a TypeError for a numeric input.
    with pytest.raises(TypeError):
        validate_prompt(123)


# Check that the maximum permitted length is accepted.
def test_maximum_prompt_length():

    # Create a prompt exactly at the character limit.
    prompt = "A" * MAX_PROMPT_CHARACTERS

    # Confirm validation accepts it.
    assert validate_prompt(prompt) == prompt


# Check that an oversized prompt is rejected.
def test_oversized_prompt():

    # Create a prompt one character over the limit.
    prompt = "A" * (MAX_PROMPT_CHARACTERS + 1)

    # Expect a ValueError for the oversized prompt.
    with pytest.raises(ValueError):
        validate_prompt(prompt)
