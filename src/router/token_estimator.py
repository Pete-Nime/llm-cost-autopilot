
# Import math to round estimated token counts upward.
import math


# Approximate the number of characters per token.
# This is only a planning heuristic, not an exact tokenizer.
CHARS_PER_TOKEN = 4


# Reserve additional tokens for message formatting.
MESSAGE_OVERHEAD_TOKENS = 20


# Estimate how many input tokens a prompt may use.
def estimate_input_tokens(prompt: str) -> int:

    # Ensure the prompt is a string.
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")

    # Reject prompts that contain only whitespace.
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Count all characters in the original prompt.
    character_count = len(prompt)

    # Estimate tokens and round upward.
    estimated_tokens = math.ceil(
        character_count / CHARS_PER_TOKEN
    )

    # Add a small allowance for message formatting.
    return estimated_tokens + MESSAGE_OVERHEAD_TOKENS
