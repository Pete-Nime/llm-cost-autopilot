
# Import sqlite3 so we can inspect the database.
import sqlite3

# Import our database initialization function.
from src.database.db import initialize_database


# Test that our database and request_logs table are created.
def test_initialize_database(tmp_path):

    # Create a temporary database location for testing.
    test_db = tmp_path / "test_autopilot.db"

    # Run our database initialization function.
    initialize_database(db_path=test_db)

    # Confirm the database file was created.
    assert test_db.exists()

    # Open the temporary SQLite database.
    with sqlite3.connect(test_db) as connection:

        # Ask SQLite whether the request_logs table exists.
        cursor = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'request_logs'
            """
        )

        # Read the first matching result.
        result = cursor.fetchone()

    # Confirm that the expected table exists.
    assert result is not None

    # Confirm the table has the correct name.
    assert result[0] == "request_logs"


# Test that database initialization can run more than once.
def test_initialize_database_twice(tmp_path):

    # Create a temporary database location.
    test_db = tmp_path / "test_autopilot.db"

    # Initialize the database for the first time.
    initialize_database(db_path=test_db)

    # Initialize it again without deleting existing tables.
    initialize_database(db_path=test_db)

    # Confirm the database still exists.
    assert test_db.exists()
