
# Import regular expressions to find words and sentences.
import re


# Define verbs that commonly describe user instructions.
INSTRUCTION_VERBS = (
    "explain",
    "compare",
    "summarize",
    "analyse",
    "analyze",
    "design",
    "implement",
    "evaluate",
    "debug",
    "optimize",
    "create",
    "write",
    "describe",
)


# Define words that may indicate task constraints.
CONSTRAINT_WORDS = (
    "must",
    "should",
    "within",
    "without",
    "budget",
    "deadline",
    "security",
    "performance",
    "scalability",
    "requirements",
)


# Define common requested output formats.
OUTPUT_FORMATS = (
    "json",
    "table",
    "csv",
    "report",
    "code",
    "markdown",
    "bullet points",
    "step-by-step",
)


# Create a function that extracts numerical features.
def extract_features(prompt: str) -> dict:

    # Ensure the supplied prompt is a string.
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    # Remove whitespace from both ends.
    cleaned_prompt = prompt.strip()

    # Reject empty prompts.
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty.")

    # Convert the prompt to lowercase.
    text = cleaned_prompt.lower()

    # Extract individual words using regular expressions.
    words = re.findall(r"\b\w+\b", text)

    # Count the number of words.
    word_count = len(words)

    # Count the number of characters.
    character_count = len(cleaned_prompt)

    # Count sentences using punctuation as an approximation.
    sentence_count = max(
        1,
        len(re.findall(r"[.!?]+", cleaned_prompt))
    )

    # Create a set for efficient word matching.
    word_set = set(words)

    # Count instruction verbs found in the prompt.
    instruction_count = sum(
        1
        for verb in INSTRUCTION_VERBS
        if verb in word_set
    )

    # Count constraint words found in the prompt.
    constraint_count = sum(
        1
        for word in CONSTRAINT_WORDS
        if word in word_set
    )

    # Count requested output formats.
    output_format_count = sum(
        1
        for output_format in OUTPUT_FORMATS
        if output_format in text
    )

    # Count context words after common context indicators.
    context_words = (
        "because",
        "background",
        "context",
        "currently",
        "existing",
    )

    # Estimate how much context is provided.
    context_length = sum(
        1
        for word in words
        if word in context_words
    )

    # Return numerical features for machine learning.
    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "instruction_count": instruction_count,
        "constraint_count": constraint_count,
        "output_format_count": output_format_count,
        "context_length": context_length,
    }
