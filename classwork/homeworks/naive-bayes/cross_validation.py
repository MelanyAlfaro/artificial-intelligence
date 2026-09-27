import random


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
