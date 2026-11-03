
# Import SQLite, Python's built-in database library.
import sqlite3

# Import Path to work with file and folder locations.
from pathlib import Path


# Find the main project folder using this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define where our database file will be stored.
DATABASE_PATH = PROJECT_ROOT / "data" / "llm_cost_autopilot.db"


# Create the database and its request logging table.
def initialize_database(db_path: Path = DATABASE_PATH) -> None:

    # Convert the supplied database location into a Path.
    db_path = Path(db_path)

    # Create the parent folder if it does not already exist.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Open a connection to the SQLite database.
    # SQLite creates the database file if it does not exist.
    with sqlite3.connect(db_path) as connection:

        # Execute SQL to create our request logging table.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS request_logs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                model_name TEXT NOT NULL,

                provider TEXT NOT NULL,

                complexity TEXT NOT NULL,

                input_tokens INTEGER NOT NULL,

                output_tokens INTEGER NOT NULL,

                latency_ms REAL NOT NULL,

                estimated_cost_usd REAL NOT NULL,

                escalated INTEGER NOT NULL DEFAULT 0,

                attempts INTEGER NOT NULL DEFAULT 1
            )
            """
        )

        # Save the database changes.
        connection.commit()
