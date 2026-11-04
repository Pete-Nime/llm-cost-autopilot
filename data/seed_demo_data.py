
# Import Path to create a separate demo database.
from pathlib import Path

# Import our existing database logging function.
from src.database.logger import log_request


# Find the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep demo data separate from actual usage data.
DEMO_DB = PROJECT_ROOT / "data" / "demo_autopilot.db"


# Define a few simulated AI provider attempts.
demo_records = [
    ("gpt-4o-mini", "openai", "simple", 100, 200, 450.0, 0.000135, False, 1),
    ("gpt-4o-mini", "openai", "simple", 150, 300, 520.0, 0.000203, False, 1),
    ("claude-haiku", "anthropic", "medium", 200, 400, 800.0, 0.002000, False, 1),
    ("claude-haiku", "anthropic", "medium", 250, 500, 900.0, 0.002500, False, 1),
    ("gpt-4o-mini", "openai", "simple", 100, 0, 400.0, 0.000015, False, 1),
    ("gpt-4o", "openai", "simple", 100, 450, 1100.0, 0.004750, True, 2),
    ("gpt-4o", "openai", "complex", 350, 800, 1500.0, 0.008875, False, 1),
]


# Run this block only when executing this file directly.
if __name__ == "__main__":

    # Avoid adding duplicate demo records on repeated runs.
    if DEMO_DB.exists():
        print("Demo database already exists. No records added.")

    else:
        # Insert each simulated provider attempt.
        for record in demo_records:

            # Save the record into our separate demo database.
            log_request(
                model_name=record[0],
                provider=record[1],
                complexity=record[2],
                input_tokens=record[3],
                output_tokens=record[4],
                latency_ms=record[5],
                estimated_cost_usd=record[6],
                escalated=record[7],
                attempts=record[8],
                db_path=DEMO_DB,
            )

        # Confirm how many records were created.
        print(f"Created {len(demo_records)} demo provider attempts.")
        print(f"Demo database: {DEMO_DB}")
