
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
    return estimate_max_workflow_cost() <= MAX_REQUEST_COST_USD


# Estimate the planned cost for a particular AI model.
def estimate_model_workflow_cost(
    model_name: str,
    input_tokens: int = MAX_ESTIMATED_INPUT_TOKENS,
    output_tokens: int = MAX_OUTPUT_TOKENS,
    attempts: int = MAX_PROVIDER_ATTEMPTS,
) -> float:

    # Reject invalid attempt counts.
    if attempts < 1:
        raise ValueError("Attempts must be at least 1")

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


# Estimate the cost of a routed workflow, including fallback.
def estimate_routed_workflow_cost(
    prompt: str,
    input_tokens: int = MAX_ESTIMATED_INPUT_TOKENS,
    output_tokens: int = MAX_OUTPUT_TOKENS,
) -> float:

    # Import our existing router to choose the first model.
    from src.router.intelligent_router import route_prompt

    # Determine which model handles the prompt.
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


# Check whether the routed workflow fits our budget.
def is_routed_workflow_within_budget(
    prompt: str,
    input_tokens: int = MAX_ESTIMATED_INPUT_TOKENS,
    output_tokens: int = MAX_OUTPUT_TOKENS,
) -> bool:

    # Estimate the total possible workflow cost.
    estimated_cost = estimate_routed_workflow_cost(
        prompt=prompt,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # Compare with the configured request budget.
    return estimated_cost <= MAX_REQUEST_COST_USD
