import random

from common import load_lyrics
from lyrics_preprocessing import preprocess_lyrics
from naive_bayes_model import classify_with_bayes, map_lyrics_to_categories


def k_fold_split(
    lyrics_by_category: dict[str, list[str]], k: int = 5, seed: int | None = None
) -> list[list[tuple[str, str]]]:
    """
    Split each category's lyrics into k randomized, stratified folds for cross-validation.

    Args:
        lyrics_by_category (dict[str, list[str]]): A dictionary mapping categories to lists of lyrics.
        k (int): The number of folds to create. Default is 5.
        seed (int | None): An optional seed for the random number generator to ensure reproducibility. Default is None.

    Returns:
        list[list[tuple[str, str]]]: A list of k folds, where each fold is a list of (category, lyric) tuples.
    """
    random_generator = random.Random(seed)
    # Initialize k empty folds
    folds: list[list[tuple[str, str]]] = [[] for _ in range(k)]

    # For each category, shuffle its lyrics and distribute them into the k folds
    for category, lyrics in lyrics_by_category.items():
        # Just a sanity check to ensure we have enough lyrics to create k folds
        if len(lyrics) < k:
            raise ValueError(
                f"Category '{category}' has only {len(lyrics)} songs, fewer than k={k}."
            )

        # Copy before shuffling, so we don't mutate the caller's list
        shuffled_lyrics = lyrics[:]
        random_generator.shuffle(shuffled_lyrics)

        fold_size = len(shuffled_lyrics) // k
        for i in range(k):
            start_index = i * fold_size
            # Ensure the last fold takes any remaining lyrics due to integer division
            end_index = start_index + fold_size if i < k - 1 else len(shuffled_lyrics)

            # For each lyric in the current slice, append it to the corresponding fold
            # keeping the structure as (category, lyric) tuples
            for single_lyric in shuffled_lyrics[start_index:end_index]:
                folds[i].append((category, single_lyric))

    return folds


# TESTING CODE
if __name__ == "__main__":
    lyrics_by_category = map_lyrics_to_categories(load_lyrics("data/training-lyrics.csv"))

    print("=== Original data ===")
    total_songs = sum(len(lyrics) for lyrics in lyrics_by_category.values())
    for category, lyrics in lyrics_by_category.items():
        proportion = len(lyrics) / total_songs
        print(f"  {category}: {len(lyrics)} songs ({proportion:.1%})")
    print(f"  TOTAL: {total_songs} songs\n")

    folds = k_fold_split(lyrics_by_category, k=5, seed=42)

    print(f"=== Created {len(folds)} folds ===\n")

    songs_seen_total = 0
    for fold_index, fold in enumerate(folds):
        print(f"--- Fold {fold_index} ({len(fold)} songs) ---")

        # Count how many songs of each category landed in this fold
        counts_in_fold: dict[str, int] = {}
        for category, _lyric in fold:
            counts_in_fold[category] = counts_in_fold.get(category, 0) + 1

        for category, count in counts_in_fold.items():
            proportion = count / len(fold)
            print(f"  {category}: {count} songs ({proportion:.1%})")

        songs_seen_total += len(fold)
        print()

    print("=== Sanity checks ===")
    print(f"Sum of all fold sizes: {songs_seen_total} (should equal original total: {total_songs})")
    print(f"Match: {songs_seen_total == total_songs}")

