
# Import our shared maximum prompt length.
from config.budget import MAX_PROMPT_CHARACTERS


# Validate and clean a prompt before routing it to an AI model.
def validate_prompt(prompt: str) -> str:

    # Reject values that are not Python strings.
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string")

    # Remove spaces and line breaks from the beginning and end.
    cleaned_prompt = prompt.strip()

    # Reject prompts that are empty after cleaning.
    if not cleaned_prompt:
        raise ValueError("Prompt cannot be empty or whitespace")

    # Reject prompts longer than the configured character limit.
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"Prompt cannot exceed {MAX_PROMPT_CHARACTERS} characters"
        )

    # Return the cleaned and validated prompt.
    return cleaned_prompt
