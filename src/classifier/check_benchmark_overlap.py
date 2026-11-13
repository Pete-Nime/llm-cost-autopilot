
# Import Python's built-in JSON library.
import json

# Import Path to locate project files.
from pathlib import Path

# Import TF-IDF to convert text into numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import cosine similarity to compare text vectors.
from sklearn.metrics.pairwise import cosine_similarity


# Find the project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the location of our training dataset.
TRAINING_PATH = PROJECT_ROOT / "data" / "complexity_dataset_v2.json"

# Define the location of our benchmark dataset.
BENCHMARK_PATH = PROJECT_ROOT / "data" / "complexity_benchmark_v1.json"


# Open and read our training dataset.
with TRAINING_PATH.open("r", encoding="utf-8") as file:
    training_data = json.load(file)

# Open and read our benchmark dataset.
with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
    benchmark_data = json.load(file)


# Extract the training questions.
training_prompts = [item["prompt"] for item in training_data]

# Extract the benchmark questions.
benchmark_prompts = [item["prompt"] for item in benchmark_data]


# Create the TF-IDF text converter.
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english",
)

# Learn vocabulary from both datasets for comparison.
all_vectors = vectorizer.fit_transform(
    training_prompts + benchmark_prompts
)

# Separate the training vectors.
training_vectors = all_vectors[:len(training_prompts)]

# Separate the benchmark vectors.
benchmark_vectors = all_vectors[len(training_prompts):]


# Compare each benchmark question against training questions.
similarities = cosine_similarity(
    benchmark_vectors,
    training_vectors,
)

# Set a threshold for flagging potentially similar questions.
THRESHOLD = 0.65

# Create an empty list for suspicious matches.
flagged_matches = []


# Check every benchmark question.
for index, benchmark_item in enumerate(benchmark_data):

    # Find the most similar training question.
    best_index = similarities[index].argmax()

    # Get its similarity score.
    best_score = float(similarities[index, best_index])

    # Flag the question if similarity exceeds our threshold.
    if best_score >= THRESHOLD:

        # Store information for manual review.
        flagged_matches.append({
            "benchmark_id": benchmark_item["id"],
            "benchmark_prompt": benchmark_item["prompt"],
            "training_prompt": training_data[best_index]["prompt"],
            "similarity": round(best_score, 4),
        })


# Print the results.
print("\n--- Benchmark Similarity Check ---")
print("Training questions:", len(training_data))
print("Benchmark questions:", len(benchmark_data))
print("Similarity threshold:", THRESHOLD)
print("Flagged matches:", len(flagged_matches))


# Display every flagged match.
for match in flagged_matches:

    # Show the benchmark question ID.
    print("\nBenchmark ID:", match["benchmark_id"])

    # Show the similarity score.
    print("Similarity:", match["similarity"])

    # Show the benchmark question.
    print("Benchmark:", match["benchmark_prompt"])

    # Show the matching training question.
    print("Training:", match["training_prompt"])


# Explain how to interpret the results.
print("\nReview flagged matches manually.")
print("Low TF-IDF similarity does not prove semantic independence.")
