
# Import JSON to read our training dataset.
import json

# Import Path to locate our dataset file.
from pathlib import Path

# Import TF-IDF to convert text into numerical features.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import cosine similarity to compare prompt text.
from sklearn.metrics.pairwise import cosine_similarity


# Define the location of our V2 dataset.
DATASET_PATH = Path("data/complexity_dataset_v2.json")


# Define a function to check dataset quality.
def check_dataset_quality():

    # Read our JSON dataset.
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    # Extract the prompt text from each record.
    prompts = [
        item["prompt"]
        for item in dataset
    ]

    # Extract the complexity labels.
    labels = [
        item["complexity"]
        for item in dataset
    ]

    # Extract the prompt IDs.
    ids = [
        item["id"]
        for item in dataset
    ]

    # Convert prompts into numerical TF-IDF vectors.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english"
    )

    # Learn vocabulary and transform our prompts.
    vectors = vectorizer.fit_transform(prompts)

    # Calculate similarities between all prompt pairs.
    similarity_matrix = cosine_similarity(vectors)

    # Set our threshold for investigating similar prompts.
    threshold = 0.65

    # Store suspicious prompt pairs.
    similar_pairs = []

    # Loop through each prompt.
    for i in range(len(prompts)):

        # Compare with later prompts only.
        for j in range(i + 1, len(prompts)):

            # Read the similarity score.
            score = similarity_matrix[i][j]

            # Flag pairs exceeding our threshold.
            if score >= threshold:

                # Store the IDs, labels and score.
                similar_pairs.append(
                    (
                        ids[i],
                        ids[j],
                        labels[i],
                        labels[j],
                        round(float(score), 3),
                        prompts[i],
                        prompts[j]
                    )
                )

    # Sort flagged pairs by similarity, highest first.
    similar_pairs.sort(
        key=lambda pair: pair[4],
        reverse=True
    )

    # Display the dataset size.
    print("\n--- V2 Dataset Quality Report ---")
    print(f"Total prompts: {len(prompts)}")

    # Display the number of flagged pairs.
    print(f"Potentially similar pairs: {len(similar_pairs)}")

    # Display the ten most similar pairs.
    for pair in similar_pairs[:10]:

        # Unpack the saved information.
        id_a, id_b, label_a, label_b, score, text_a, text_b = pair

        # Display the prompt identifiers and score.
        print(
            f"\n{id_a} ({label_a}) vs "
            f"{id_b} ({label_b})"
        )

        # Display the similarity score.
        print(f"Similarity: {score}")

        # Display the first prompt.
        print(f"A: {text_a}")

        # Display the second prompt.
        print(f"B: {text_b}")

    # Display a message if no pairs were flagged.
    if not similar_pairs:
        print("\nNo pairs exceeded the similarity threshold.")


# Run our checker when this file is executed directly.
if __name__ == "__main__":
    check_dataset_quality()
