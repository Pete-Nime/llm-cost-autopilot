
# Import JSON to read and save datasets.
import json

# Import sys to read a filename from the terminal.
import sys

# Import Counter to count labels.
from collections import Counter

# Import Path to work with file locations.
from pathlib import Path


# Define the location of our main dataset.
MAIN_PATH = Path("data/complexity_dataset_v2.json")

# Define the allowed complexity labels.
VALID_LABELS = {"simple", "medium", "complex"}


# Create a function to validate a collection of prompts.
def check_records(records):

    # Check that the dataset is a list.
    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON list.")

    # Create sets to detect duplicate IDs and text.
    seen_ids = set()
    seen_prompts = set()

    # Check every record.
    for index, item in enumerate(records, start=1):

        # Check that the record is a dictionary.
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not an object.")

        # Check that the required fields exist.
        if not {"id", "prompt", "complexity"}.issubset(item):
            raise ValueError(f"Missing fields in record {index}.")

        # Read the record values.
        prompt_id = item["id"]
        prompt = item["prompt"]
        label = item["complexity"]

        # Check that the ID is valid.
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"Invalid ID at record {index}.")

        # Check that the prompt contains text.
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Invalid prompt at record {index}.")

        # Check that the label is allowed.
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label at record {index}.")

        # Normalize the prompt for duplicate checking.
        normalized = " ".join(prompt.casefold().split())

        # Reject repeated IDs.
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate ID: {prompt_id}")

        # Reject repeated prompt text.
        if normalized in seen_prompts:
            raise ValueError(f"Duplicate prompt: {prompt_id}")

        # Remember the current ID and prompt.
        seen_ids.add(prompt_id)
        seen_prompts.add(normalized)


# Create a function to merge a new batch.
def merge_batch(batch_path):

    # Read our existing dataset.
    main_data = json.loads(
        MAIN_PATH.read_text(encoding="utf-8")
    )

    # Read the new batch.
    batch_data = json.loads(
        batch_path.read_text(encoding="utf-8")
    )

    # Validate each input and the combined dataset.
    check_records(main_data)
    check_records(batch_data)

    # Combine the two collections in memory.
    combined = main_data + batch_data

    # Check for duplicates across both files.
    check_records(combined)

    # Count the prompts in each category.
    counts = Counter(
        item["complexity"]
        for item in combined
    )

    # Create a backup filename based on the existing size.
    backup_path = Path(
        f"data/complexity_dataset_v2_backup_{len(main_data)}.json"
    )

    # Refuse to overwrite an existing backup.
    if backup_path.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_path}"
        )

    # Save the existing dataset before modifying it.
    backup_path.write_text(
        json.dumps(main_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # Save the merged dataset.
    MAIN_PATH.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    # Print the merge results.
    print("\n--- Dataset Merge Successful ---")
    print(f"Total prompts: {len(combined)}")

    # Display each category count.
    for label in ["simple", "medium", "complex"]:
        print(f"{label}: {counts[label]}")

    # Display the backup and updated file paths.
    print(f"Backup: {backup_path}")
    print(f"Updated: {MAIN_PATH}")


# Execute the script when called from the terminal.
if __name__ == "__main__":

    # Require exactly one batch filename.
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.classifier.merge_dataset "
            "data/your_batch.json"
        )

    # Convert the filename into a Path object.
    batch_path = Path(sys.argv[1])

    # Check that the supplied batch file exists.
    if not batch_path.is_file():
        raise SystemExit(f"Batch not found: {batch_path}")

    # Merge the supplied batch.
    merge_batch(batch_path)
