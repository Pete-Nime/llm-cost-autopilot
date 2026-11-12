
# Import JSON to read our labelled dataset.
import json

# Import CSV to save a reviewable spreadsheet-style report.
import csv

# Import Path to work with project file locations.
from pathlib import Path


# Define the location of our V2 dataset.
DATASET_PATH = Path("data/complexity_dataset_v2.json")

# Define the location of our review report.
REPORT_PATH = Path("reports/complexity_label_review.csv")


# Define the main function for our review script.
def main():

    # Read the dataset from its JSON file.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Create the reports folder if it does not exist.
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Open a CSV file for our review results.
    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        # Define the column headings.
        fieldnames = [
            "id",
            "prompt",
            "current_label",
            "word_count",
            "reviewed_label",
            "review_notes"
        ]

        # Create a CSV writer.
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        # Write the column headings.
        writer.writeheader()

        # Go through every prompt.
        for item in dataset:

            # Count the words in the prompt.
            word_count = len(
                item["prompt"].split()
            )

            # Write one row for human review.
            writer.writerow({
                "id": item["id"],
                "prompt": item["prompt"],
                "current_label": item["complexity"],
                "word_count": word_count,
                "reviewed_label": "",
                "review_notes": ""
            })

    # Display a success message.
    print("\n--- Dataset Label Review ---")
    print(f"Prompts exported: {len(dataset)}")
    print(f"Report created: {REPORT_PATH}")
    print("Dataset JSON remains unchanged.")


# Run our function when this file is executed directly.
if __name__ == "__main__":
    main()
