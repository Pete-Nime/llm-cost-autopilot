
# Import Literal to define our three valid complexity labels.
from typing import Literal

# Define the labels our classifier is allowed to return.
Complexity = Literal["simple", "medium", "complex"]


# Define phrases that often indicate complex work.
COMPLEX_PHRASES = (
    "design a system",
    "architect",
    "fault-tolerant",
    "production-ready",
    "step-by-step implementation",
    "trade-offs",
    "optimize",
    "debug",
    "production database",
    "peak traffic",
    "recurring outages",
    "migrate",
    "migration",
    "evaluate whether",
    "performance, cost",
    "scalability",
    "security and maintenance",
    "root cause",
    "prevent future",
)


# Define phrases that often indicate medium difficulty.
MEDIUM_PHRASES = (
    "explain how",
    "compare",
    "difference between",
    "summarize",
    "write a python function",
    "give an example",
    "describe how",
    "why does",
    "why do",
    "walk me through",
    "how does",
    "how do",
    "stages of",
    "advantages and disadvantages",
)


# Define phrases that often indicate simple questions.
SIMPLE_PHRASES = (
    "what is",
    "what does",
    "define",
    "basic definition",
    "what are",
    "what is the purpose",
)


# Create a function that predicts question complexity.
def classify_complexity(prompt: str) -> Complexity:

    # Check that the input is a string.
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    # Remove unnecessary whitespace.
    cleaned_prompt = prompt.strip()

    # Reject empty questions.
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty.")

    # Convert the question to lowercase.
    text = cleaned_prompt.lower()

    # Check complex patterns first because they require more care.
    if any(phrase in text for phrase in COMPLEX_PHRASES):
        return "complex"

    # Next, check for medium complexity patterns.
    if any(phrase in text for phrase in MEDIUM_PHRASES):
        return "medium"

    # Check whether the question matches a simple pattern.
    if any(phrase in text for phrase in SIMPLE_PHRASES):
        return "simple"

    # Use medium when we cannot confidently identify difficulty.
    # This is a cautious fallback, not proof the task is medium.
    return "medium"
