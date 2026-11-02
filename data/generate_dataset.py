
# Import json to save our dataset in JSON format.
import json

# Import Path to work with file locations.
from pathlib import Path

# Define the topics our questions will cover.
topics = [
    "Python", "SQL", "machine learning", "artificial intelligence",
    "data engineering", "ETL pipelines", "cloud computing",
    "REST APIs", "databases", "unit testing",
    "Docker", "Git", "cybersecurity", "data privacy"
]

# Define five simple question patterns.
simple_templates = [
    "What is {}?",
    "Define {}.",
    "What does {} mean?",
    "Give a basic definition of {}.",
    "What is the purpose of {}?"
]

# Define five medium question patterns.
medium_templates = [
    "Explain how {} works.",
    "Compare {} with related technologies.",
    "Describe how {} is used in a real-world project.",
    "Give an example of {} in practice.",
    "Summarize the main benefits of {}."
]

# Define five complex question patterns.
complex_templates = [
    "Design a system that uses {} at enterprise scale.",
    "Architect a production-ready solution involving {}.",
    "Optimize a large-scale implementation of {}.",
    "Debug a complex production failure involving {}.",
    "Analyze the trade-offs of different {} architectures."
]

# Create an empty list to store our questions.
dataset = []

# Generate 70 simple questions.
for topic in topics:
    for template in simple_templates:
        dataset.append({
            "id": f"P{len(dataset) + 1:03d}",
            "prompt": template.format(topic),
            "complexity": "simple"
        })

# Generate 70 medium questions.
for topic in topics:
    for template in medium_templates:
        dataset.append({
            "id": f"P{len(dataset) + 1:03d}",
            "prompt": template.format(topic),
            "complexity": "medium"
        })

# Generate 70 complex questions.
for topic in topics:
    for template in complex_templates:
        dataset.append({
            "id": f"P{len(dataset) + 1:03d}",
            "prompt": template.format(topic),
            "complexity": "complex"
        })

# Find the data folder containing this script.
data_folder = Path(__file__).resolve().parent

# Choose the output JSON file.
output_file = data_folder / "complexity_dataset.json"

# Save the dataset in a readable JSON format.
output_file.write_text(
    json.dumps(dataset, indent=2),
    encoding="utf-8"
)

# Display the number of generated questions.
print(f"Generated {len(dataset)} labelled prompts.")

# Display where the dataset was saved.
print(f"Saved to: {output_file}")
