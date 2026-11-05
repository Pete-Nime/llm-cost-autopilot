
# Import the budget configuration values.
from config.budget import (
    MAX_REQUEST_COST_USD,
    MAX_ESTIMATED_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MAX_PROVIDER_ATTEMPTS,
    PLANNING_INPUT_USD_PER_MILLION,
    PLANNING_OUTPUT_USD_PER_MILLION,
)


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
