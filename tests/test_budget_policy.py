
# Import our cost estimation and budget policy functions.
from src.router.budget_policy import (
    estimate_max_workflow_cost,
    is_workflow_within_budget,
)


# Verify our planning estimate is calculated correctly.
def test_estimated_workflow_cost():

    # Expected cost with the current example configuration.
    expected_cost = 0.019

    # Compare floating-point numbers with a small tolerance.
    assert abs(estimate_max_workflow_cost() - expected_cost) < 1e-10


# Verify the configured budget rejects this estimate.
def test_workflow_exceeds_budget():

    # The estimated cost is above our $0.01 threshold.
    assert is_workflow_within_budget() is False
