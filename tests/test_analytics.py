
# Import our existing function for saving request records.
from src.database.logger import log_request

# Import the two new analytics functions.
from src.database.analytics import (
    get_usage_summary,
    get_model_breakdown,
)


# Test our overall usage calculations.
def test_usage_summary(tmp_path):

    # Use a temporary database to protect real data.
    test_db = tmp_path / "analytics_test.db"

    # Save the first simulated provider attempt.
    log_request(
        model_name="gpt-4o-mini",
        provider="openai",
        complexity="simple",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
        estimated_cost_usd=0.001,
        escalated=False,
        attempts=1,
        db_path=test_db,
    )

    # Save a second simulated provider attempt.
    log_request(
        model_name="gpt-4o",
        provider="openai",
        complexity="simple",
        input_tokens=15,
        output_tokens=30,
        latency_ms=300.0,
        estimated_cost_usd=0.003,
        escalated=True,
        attempts=2,
        db_path=test_db,
    )

    # Calculate the summary from our stored records.
    summary = get_usage_summary(db_path=test_db)

    # Confirm two provider attempts were recorded.
    assert summary["total_attempts"] == 2

    # Confirm total input tokens: 10 + 15 = 25.
    assert summary["total_input_tokens"] == 25

    # Confirm total output tokens: 20 + 30 = 50.
    assert summary["total_output_tokens"] == 50

    # Confirm the combined estimated cost.
    assert abs(summary["total_cost_usd"] - 0.004) < 1e-10

    # Confirm average latency: (100 + 300) / 2.
    assert summary["average_latency_ms"] == 200.0

    # Confirm that one fallback attempt was recorded.
    assert summary["fallback_attempts"] == 1


# Test spending grouped by AI model.
def test_model_breakdown(tmp_path):

    # Create a temporary test database.
    test_db = tmp_path / "models_test.db"

    # Record usage for our first model.
    log_request(
        model_name="gpt-4o-mini",
        provider="openai",
        complexity="simple",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
        estimated_cost_usd=0.001,
        db_path=test_db,
    )

    # Record usage for our second model.
    log_request(
        model_name="gpt-4o",
        provider="openai",
        complexity="complex",
        input_tokens=15,
        output_tokens=30,
        latency_ms=200.0,
        estimated_cost_usd=0.005,
        db_path=test_db,
    )

    # Get the spending breakdown.
    breakdown = get_model_breakdown(db_path=test_db)

    # Confirm we have two different models.
    assert len(breakdown) == 2

    # The most expensive model should appear first.
    assert breakdown[0]["model_name"] == "gpt-4o"

    # Confirm the highest model cost.
    assert abs(breakdown[0]["total_cost_usd"] - 0.005) < 1e-10


# Test that analytics also work on an empty database.
def test_empty_database_summary(tmp_path):

    # Choose a database that does not exist yet.
    test_db = tmp_path / "empty_test.db"

    # Our analytics function should initialize it safely.
    summary = get_usage_summary(db_path=test_db)

    # No AI attempts have been logged.
    assert summary["total_attempts"] == 0

    # No estimated costs have been incurred.
    assert summary["total_cost_usd"] == 0
