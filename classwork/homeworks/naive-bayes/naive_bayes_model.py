from collections import defaultdict

def map_lyrics_to_categories(
  lyrics: list[tuple[str, str]]
) -> dict[str, list[str]]:
  """Converts the list of tupled category, lyric pairs into a dictionary of category to list of lyrics

  Args:
    lyrics (list[tuple[str, str]]): List of tupled category, lyric pairs

  Returns:
    dict[str, list[str]]: The dictionary mapping categories to lyrics directly
  """
  category_to_lyrics = defaultdict(list)

  for category, lyric in lyrics:
    category_to_lyrics[category].append(lyric)

  return dict(category_to_lyrics)


def calculate_priors(
    category_to_lyrics: dict[str, list[str]]
) -> dict[str, float]:
    """Calculates the prior probability for each category

    Args:
      category_to_lyrics (dict[str, list[str]]): Dict associating category names to lyrics list 

    Returns:
      dict[str, float]: Dict associating categories with prior probabilities
    """
    # All lyrics in the set
    total_lyrics = sum(len(lyrics) for lyrics in category_to_lyrics.values())

    priors: dict[str, float] = {}

    for category, lyrics in category_to_lyrics.items():
        priors[category] = len(lyrics) / total_lyrics

    return priors


def remove_unknown_words(test_set: str, vocabulary: set) -> str:
  """Removes unknown words from a test set

  Args:
      test_set (str): The test set to clean
      vocabulary (set): The words in the corpus

  Returns:
      str: The test set without unknown words
  """
  # Split the test set by any whitespaces into the words
  test_set_words = test_set.split()

  cleaned_test_set = []

  # Only leave known words in the clean test set
  for word in test_set_words:
    if word in vocabulary:
      cleaned_test_set.append(word)

  # Reconstruct into string
  return " ".join(cleaned_test_set)


def count_words_per_category(category_to_lyrics: dict[str, list[str]]) -> dict[str, int]:
  """Counts how many words are in each category

  Args:
    category_to_lyrics (dict[str, list[str]]): Dict associating categories to their lyrics

  Returns:
    dict[str, int]: Categories associated to the amount of words they have
  """
  category_to_word_counts: dict[str, int] = dict()
  
  # For each category
  for category, lyrics in category_to_lyrics.items():
    category_to_word_counts[category] = 0

    # For each lyric row, add to counter how many words there are
    for lyric in lyrics:
      category_to_word_counts[category] += len(lyric.split())
    
  return category_to_word_counts
