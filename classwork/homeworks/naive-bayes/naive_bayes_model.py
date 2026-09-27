import math
from collections import defaultdict
from vocabulary import extract_vocabulary
from lyrics_preprocessing import create_bag_of_words
from common import Category

class NaiveBayesModel:
  def __init__(
    self,
    training_sets: list[tuple[Category, str]]
  ):
    # {word: {category: probability}}
    cat_to_training_sets = self._map_training_sets_to_categories(training_sets)
    bags: dict[Category, dict[str, int]] = create_bag_of_words(training_sets)

    self._vocabulary: set[str] = extract_vocabulary(bags)
    self._likelihood_logs: dict[str, dict[Category, float]] = defaultdict(dict)
    self._categories: set[Category] = set(bags.keys())
    self._priors: dict[Category, float] = self._calculate_priors(cat_to_training_sets)
    self._train(bags)


  def _train(self, bags: dict[Category, dict[str, int]]) -> None:
    """
    Train the model, creating the likelihoods with laplace smooting.

    Args:
      bags: The bags containing the words asociated to the categories.
    """
    cats = bags.keys()
    category_to_word_counts: dict[str, int] = self._count_words_per_category(bags)
    for word in self._vocabulary:
      for cat in cats:
        # Save the likelihoods as logarithms for faster and more acurate
        # calculations
        self._likelihood_logs[word][cat] = math.log(
          self._likelihood_laplace_smoothing(
            word,
            bags[cat],
            category_to_word_counts[cat],
            len(self._vocabulary)))


  def classify(self, test_set: str) -> Category:
    """
    Classify a test set into a category.

    Args:
      test_set: The test set to classify.

    Returns:
      Category: The category that the test set was classified as.
    """
    processed_test_set = self._process_test_set(test_set)
    scores: dict[Category, float] = {}
    # Calculate the score per category
    # Use logarithms to keep the numbers small and work with addition
    for cat in self._categories:
      scores[cat] = math.log(self._priors[cat])
      for word in processed_test_set:
        scores[cat] += self._likelihood_logs[word][cat]

    # Iterate over scores (keys) and compare based on the value
    # (using the method 'get')
    return max(iterable=scores, key=scores.get)


  def _process_test_set(self, test_set: str) -> list[str]:
    """Removes unknown words from a test set and returns as separated list of words

    Args:
        test_set (str): The test set to clean

    Returns:
        list[str]: The test set without unknown words, separated by words in a list
    """
    # Split the test set by any whitespaces into the words
    test_set_words = test_set.split()

    processed_test_set = []

    # Only leave known words in the clean test set
    for word in test_set_words:
      if word in self._vocabulary:
        processed_test_set.append(word)

    # Reconstruct into string
    return processed_test_set

  def _likelihood_laplace_smoothing(
    self,
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


  def _count_words_per_category(self, bags: dict[str, dict[str, int]]) -> dict[str, int]:
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


  def _calculate_priors(
    self,
    category_to_training_sets: dict[str, list[str]]
  ) -> dict[str, float]:
      """Calculates the prior probability for each category

      Args:
        category_to_training_sets (dict[str, list[str]]): Dict associating category names to training sets list 

      Returns:
        dict[str, float]: Dict associating categories with prior probabilities
      """
      # All training_sets in the set
      total_training_sets = sum(len(training_sets) for training_sets in category_to_training_sets.values())

      priors: dict[str, float] = {}

      for category, training_sets in category_to_training_sets.items():
          priors[category] = len(training_sets) / total_training_sets

      return priors


  def _map_training_sets_to_categories(
    self,
    training_sets: list[tuple[str, str]]
  ) -> dict[str, list[str]]:
    """Converts the list of tupled category, training set pairs into a dictionary of category to list of training sets

    Args:
      training_sets (list[tuple[str, str]]): List of tupled category, training pairs

    Returns:
      dict[str, list[str]]: The dictionary mapping categories to training sets directly
    """
    category_to_training_sets = defaultdict(list)

    for category, training_set in training_sets:
      category_to_training_sets[category].append(training_set)

    return dict(category_to_training_sets)
