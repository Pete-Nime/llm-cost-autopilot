
# Import SQLite so we can inspect stored records.
import sqlite3

# Import our new logging function.
from src.database.logger import log_request


# Check that one AI request can be saved successfully.
def test_log_request(tmp_path):

    # Create a temporary database for this test.
    test_db = tmp_path / "test_logs.db"

    # Save a simulated AI request.
    request_id = log_request(
        model_name="gpt-4o-mini",
        provider="openai",
        complexity="simple",
        input_tokens=15,
        output_tokens=40,
        latency_ms=1250.5,
        estimated_cost_usd=0.00003,
        escalated=False,
        attempts=1,
        db_path=test_db,
    )

    # Confirm SQLite assigned a record ID.
    assert request_id == 1

    # Open the temporary database.
    with sqlite3.connect(test_db) as connection:

        # Retrieve the saved request.
        row = connection.execute(
            """
            SELECT
                model_name,
                provider,
                complexity,
                input_tokens,
                output_tokens,
                escalated,
                attempts
            FROM request_logs
            WHERE id = ?
            """,
            (request_id,),
        ).fetchone()

    # Confirm the saved model is correct.
    assert row[0] == "gpt-4o-mini"

    # Confirm the provider is correct.
    assert row[1] == "openai"

    # Confirm the complexity is correct.
    assert row[2] == "simple"

    # Confirm the token counts are correct.
    assert row[3] == 15
    assert row[4] == 40

    # Confirm escalation was not required.
    assert row[5] == 0

    # Confirm only one attempt occurred.
    assert row[6] == 1


# Check that an escalated request can be recorded.
def test_log_escalated_request(tmp_path):

    # Create another temporary database.
    test_db = tmp_path / "test_escalation_logs.db"

    # Save a simulated escalated request.
    request_id = log_request(
        model_name="gpt-4o",
        provider="openai",
        complexity="medium",
        input_tokens=20,
        output_tokens=80,
        latency_ms=2200.0,
        estimated_cost_usd=0.001,
        escalated=True,
        attempts=2,
        db_path=test_db,
    )

    # Open the temporary database.
    with sqlite3.connect(test_db) as connection:

        # Read the escalation flag and number of attempts.
        row = connection.execute(
            """
            SELECT escalated, attempts
            FROM request_logs
            WHERE id = ?
            """,
            (request_id,),
        ).fetchone()

    # Confirm the escalation flag was saved as 1.
    assert row[0] == 1

    # Confirm the request used two attempts.
    assert row[1] == 2
