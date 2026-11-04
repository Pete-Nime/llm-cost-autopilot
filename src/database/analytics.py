
# Import SQLite to read our stored AI usage records.
import sqlite3

# Import Path to work with database locations.
from pathlib import Path

# Import our existing database setup and default path.
from src.database.db import DATABASE_PATH, initialize_database


# Read the overall AI usage statistics.
def get_usage_summary(db_path: Path = DATABASE_PATH) -> dict:

    # Ensure the database and its table exist.
    initialize_database(db_path=db_path)

    # Connect to our SQLite database.
    with sqlite3.connect(db_path) as connection:

        # Calculate totals from every stored provider attempt.
        row = connection.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(input_tokens), 0),
                COALESCE(SUM(output_tokens), 0),
                COALESCE(SUM(estimated_cost_usd), 0),
                COALESCE(AVG(latency_ms), 0),
                COALESCE(SUM(escalated), 0)
            FROM request_logs
            """
        ).fetchone()

    # Return the statistics with readable names.
    return {
        "total_attempts": row[0],
        "total_input_tokens": row[1],
        "total_output_tokens": row[2],
        "total_cost_usd": row[3],
        "average_latency_ms": row[4],
        "fallback_attempts": row[5],
    }


# Calculate spending and usage for each AI model.
def get_model_breakdown(db_path: Path = DATABASE_PATH) -> list[dict]:

    # Ensure the database exists.
    initialize_database(db_path=db_path)

    # Open the SQLite database.
    with sqlite3.connect(db_path) as connection:

        # Group stored attempts by model name.
        rows = connection.execute(
            """
            SELECT
                model_name,
                COUNT(*),
                COALESCE(SUM(estimated_cost_usd), 0),
                COALESCE(AVG(latency_ms), 0)
            FROM request_logs
            GROUP BY model_name
            ORDER BY SUM(estimated_cost_usd) DESC
            """
        ).fetchall()

    # Convert each database row into a readable dictionary.
    return [
        {
            "model_name": row[0],
            "attempts": row[1],
            "total_cost_usd": row[2],
            "average_latency_ms": row[3],
        }
        for row in rows
    ]
