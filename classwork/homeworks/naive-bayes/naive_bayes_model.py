import math
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


def process_test_set(test_set: str, vocabulary: set) -> list[str]:
  """Removes unknown words from a test set and returns as separated list of words

  Args:
      test_set (str): The test set to clean
      vocabulary (set): The words in the corpus

  Returns:
      list[str]: The test set without unknown words, separated by words in a list
  """
  # Split the test set by any whitespaces into the words
  test_set_words = test_set.split()

  processed_test_set = []

  # Only leave known words in the clean test set
  for word in test_set_words:
    if word in vocabulary:
      processed_test_set.append(word)

  # Reconstruct into string
  return processed_test_set


def count_words_per_category(bags: dict[str, dict[str, int]]) -> dict[str, int]:
  """Counts how many words are in each category

  Args:
    bags (dict[str, dict[str, int]]): Dict associating categories to counts per word

  Returns:
    dict[str, int]: Categories associated to the amount of words they have
  """
  category_to_word_counts: dict[str, int] = {}
  
  # For each category
  for category, bag in bags.items():
    category_to_word_counts[category] = 0
    
    # Sum the counts per word
    for count in bag.values():
      category_to_word_counts[category] += count
    
  return category_to_word_counts


def likelihood_laplace_smoothing(
  word: str,
  bag: dict,
  total_words_in_class: int,
  vocabulary_size: int
) -> float:
    """Calculates the probability of a word given a category using Laplace smoothing.

    Args:
      word (str): The word to calculate the probability for.
      bag (dict): The word-count dictionary for a specific category.
      total_words_in_class (int): The total number of words in that category.
      vocabulary_size (int): The size of the vocabulary.

    Returns:
        float: The smoothed probability of the word given the category.
    """
    word_count_in_class = bag.get(word, 0)
    return (word_count_in_class + 1) / (total_words_in_class + vocabulary_size)
  
  
def classify_with_bayes(
  vocabulary: set, 
  bags: dict[str, dict[str, int]],
  category_to_lyrics: dict[str, list[str]],
  test_set: str
) -> tuple[str, float]:
  """Classifies the category that a test set belongs to using Naive Bayes with Laplace Smoothing

  Args:
    vocabulary (set): The unique words in the corpus
    bags (dict[str, dict[str, int]]): Categories associated to counts of appearances by words 
    category_to_lyrics: dict[str, list[str]]: Categories associated with the lyrics that belong to it
    test_set (str): The sentence to classify. Must be normalized before passing to the function

  Returns:
    tuple[str, float]: Name of the class that the test set is classified to and the calculated probability (log score)
  """
  # First remove unknown words to clean
  processed_test_set: list[str] = process_test_set(test_set, vocabulary)
  
  # Stores the likelihood of the test set per category for final decision
  category_to_score: dict[str, float] = {}
    
  # Calculate prior probabilities per category
  priors: dict[str, float] = calculate_priors(category_to_lyrics)
  
  # Obtain the word counts per category using bags
  category_to_word_counts: dict[str, int] = count_words_per_category(bags)
  
  # Calculate likelihoods of the test set per category
  for current_category, current_bag in bags.items():
    current_prior = priors[current_category]
    current_word_count = category_to_word_counts[current_category]
    
    # Start likelihood with log of current prior to sum other log of probabilities
    current_likelihood = math.log(current_prior)
    
    for word in processed_test_set:
      current_likelihood += math.log(likelihood_laplace_smoothing(word, current_bag, current_word_count, len(vocabulary)))
    
    # Store final likelihood for current category
    category_to_score[current_category] = current_likelihood

  # Select category with highest likelihood
  classified_category = max(
    category_to_score,
    key=category_to_score.get
  )

  return classified_category, category_to_score[classified_category]
