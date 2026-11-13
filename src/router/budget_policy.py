
# Import the budget configuration values.
from config.budget import (
    MAX_REQUEST_COST_USD,
    MAX_ESTIMATED_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_PROVIDER_ATTEMPTS,
    PLANNING_INPUT_USD_PER_MILLION,
    PLANNING_OUTPUT_USD_PER_MILLION,
)
# Import the model-specific estimator created earlier.
from src.router.model_cost_estimator import estimate_model_cost

# Import our new prompt-based token estimator.
from src.router.token_estimator import estimate_input_tokens

# Import the budget module itself so we read its current values.
import config.budget as budget_config


# Check that a token or attempt count is a valid whole number.
def validate_count(value: int, name: str, minimum: int = 0) -> None:

    # Reject decimals, strings, booleans, and other invalid types.
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")

    # Reject values below the permitted minimum.
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


# Calculate our conservative planning cost.
def estimate_max_workflow_cost() -> float:

    # Calculate the planned input cost for one provider attempt.
    input_cost = (
        MAX_ESTIMATED_INPUT_TOKENS
        * PLANNING_INPUT_USD_PER_MILLION
        / 1_000_000
    )

    # Calculate the planned output cost for one provider attempt.
    output_cost = (
        MAX_OUTPUT_TOKENS
        * PLANNING_OUTPUT_USD_PER_MILLION
        / 1_000_000
    )

    # Multiply by the maximum number of provider attempts.
    return (
        input_cost + output_cost
    ) * MAX_PROVIDER_ATTEMPTS


# Determine whether our planned workflow fits the budget.
def is_workflow_within_budget() -> bool:

    # Compare the estimated cost with the configured threshold.
    return estimate_max_workflow_cost() <= budget_config.MAX_REQUEST_COST_USD


# Estimate the planned cost for a particular AI model.
def estimate_model_workflow_cost(
    model_name: str,
    input_tokens: int = MAX_ESTIMATED_INPUT_TOKENS,
    output_tokens: int = MAX_OUTPUT_TOKENS,
    attempts: int = MAX_PROVIDER_ATTEMPTS,
) -> float:

     # Ensure input tokens are valid whole numbers.
    validate_count(input_tokens, "Input tokens")

    # Ensure output tokens are valid whole numbers.
    validate_count(output_tokens, "Output tokens")

    # Ensure at least one whole provider attempt is requested.
    validate_count(attempts, "Attempts", minimum=1)

    # Calculate the configured cost of one model call.
    cost_per_attempt = estimate_model_cost(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # Estimate the cost of repeating that model call.
    return cost_per_attempt * attempts


# Check whether a model's planned cost fits our budget.
def is_model_within_budget(
    model_name: str,
    input_tokens: int = MAX_ESTIMATED_INPUT_TOKENS,
    output_tokens: int = MAX_OUTPUT_TOKENS,
    attempts: int = MAX_PROVIDER_ATTEMPTS,
) -> bool:

    # Calculate the model-specific estimated workflow cost.
    estimated_cost = estimate_model_workflow_cost(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempts=attempts,
    )

    # Return True if the estimated cost fits the budget.
    return estimated_cost <= MAX_REQUEST_COST_USD


# Estimate the cost using an optional preselected routing decision.
def estimate_routed_workflow_cost(
    prompt: str,
    input_tokens: int | None = None,
    output_tokens: int = MAX_OUTPUT_TOKENS,
    decision=None,
) -> float:

    # Import our existing router to choose the first model.
    from src.router.intelligent_router import route_prompt

    # Estimate tokens from the prompt when no count was supplied.
    if input_tokens is None:
        input_tokens = estimate_input_tokens(prompt)

    # Validate the input-token count.
    validate_count(input_tokens, "Input tokens")

    # Validate the output-token count.
    validate_count(output_tokens, "Output tokens")
    
    # Reuse an existing routing decision when one is supplied.
    if decision is None:

        # Otherwise select the model using the original router.
        decision = route_prompt(prompt)

    # Estimate the first model's cost.
    first_cost = estimate_model_cost(
        model_name=decision.model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # GPT-4o is already the strongest fallback model.
    # It will not escalate again.
    if decision.model_name == "gpt-4o":
        return first_cost

    # Estimate the possible GPT-4o fallback cost.
    fallback_cost = estimate_model_cost(
        model_name="gpt-4o",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # Reserve enough budget for both possible attempts.
    return first_cost + fallback_cost


# Check the budget using an optional preselected decision.
def is_routed_workflow_within_budget(
    prompt: str,
    input_tokens: int | None = None,
    output_tokens: int = MAX_OUTPUT_TOKENS,
    decision=None,
) -> bool:

    # Calculate cost for the exact model we plan to execute.
    estimated_cost = estimate_routed_workflow_cost(
        prompt=prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        decision=decision,
    )

    # Compare with the configured request budget.
    return estimated_cost <= budget_config.MAX_REQUEST_COST_USD
