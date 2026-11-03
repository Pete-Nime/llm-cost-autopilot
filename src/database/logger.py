
# Import SQLite to save information in our database.
import sqlite3

# Import Path to work with database file locations.
from pathlib import Path

# Import our existing database setup function and path.
from src.database.db import (
    DATABASE_PATH,
    initialize_database,
)


# Save one completed AI request into our database.
def log_request(
    model_name: str,
    provider: str,
    complexity: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    estimated_cost_usd: float,
    escalated: bool = False,
    attempts: int = 1,
    db_path: Path = DATABASE_PATH,
) -> int:

    # Make sure the database and table exist.
    initialize_database(db_path=db_path)

    # Connect to our SQLite database.
    with sqlite3.connect(db_path) as connection:

        # Insert the request details into our table.
        cursor = connection.execute(
            """
            INSERT INTO request_logs (
                model_name,
                provider,
                complexity,
                input_tokens,
                output_tokens,
                latency_ms,
                estimated_cost_usd,
                escalated,
                attempts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                provider,
                complexity,
                input_tokens,
                output_tokens,
                latency_ms,
                estimated_cost_usd,
                int(escalated),
                attempts,
            ),
        )

        # Save the new record permanently.
        connection.commit()

        # Return the unique ID assigned to this record.
        return cursor.lastrowid
