
# Import JSON to read our dataset.
import json

# Import Counter to count complexity labels.
from collections import Counter

# Import Path to locate our dataset.
from pathlib import Path


# Define the dataset location.
DATASET_PATH = Path("data/complexity_dataset_v2.json")

# Define the allowed complexity labels.
VALID_LABELS = {"simple", "medium", "complex"}


# Create a reusable validation function.
def validate_dataset():

    # Read the JSON file.
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    # Confirm that the dataset contains a list.
    if not isinstance(dataset, list):
        raise ValueError("Dataset must contain a JSON list.")

    # Create empty collections for checking duplicates.
    ids = set()
    prompts = set()

    # Create a counter for our labels.
    counts = Counter()

    # Check each record individually.
    for index, item in enumerate(dataset, start=1):

        # Confirm that each record is a dictionary.
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not an object.")

        # Confirm that the required fields exist.
        required = {"id", "prompt", "complexity"}

        # Find any missing fields.
        missing = required - set(item)

        # Stop if fields are missing.
        if missing:
            raise ValueError(
                f"Record {index} missing fields: {missing}"
            )

        # Read the prompt ID.
        prompt_id = item["id"]

        # Read the prompt text.
        prompt_text = item["prompt"]

        # Read the complexity label.
        label = item["complexity"]

        # Validate the ID.
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"Invalid ID at record {index}.")

        # Validate the prompt text.
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"Empty prompt at record {index}.")

        # Validate the complexity label.
        if label not in VALID_LABELS:
            raise ValueError(
                f"Invalid complexity label at record {index}."
            )

        # Normalize the text for duplicate detection.
        normalized_text = " ".join(
            prompt_text.casefold().split()
        )

        # Check whether the ID already exists.
        if prompt_id in ids:
            raise ValueError(f"Duplicate ID: {prompt_id}")

        # Check whether the prompt already exists.
        if normalized_text in prompts:
            raise ValueError(
                f"Duplicate prompt: {prompt_id}"
            )

        # Remember the ID.
        ids.add(prompt_id)

        # Remember the normalized prompt.
        prompts.add(normalized_text)

        # Count the label.
        counts[label] += 1

    # Print the validation results.
    print("\n--- Dataset Validation Report ---")

    # Display the total number of prompts.
    print(f"Total prompts: {len(dataset)}")

    # Display the counts for all categories.
    for label in ["simple", "medium", "complex"]:
        print(f"{label}: {counts[label]}")

    # Confirm successful structural validation.
    print("Duplicate IDs: 0")
    print("Duplicate prompts: 0")
    print("Structural validation: PASSED")


# Execute the function when running this file directly.
if __name__ == "__main__":
    validate_dataset()
