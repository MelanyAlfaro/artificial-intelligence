import random
from collections import defaultdict

from classifiers_evaluation import Metric, evaluate_bayes_model
from common import Category, load_lyrics, map_training_sets_to_categories
from naive_bayes_model import NaiveBayesModel


def k_fold_split(
  lyrics_by_category: dict[Category, list[str]], k: int = 5, seed: int | None = None
) -> list[list[tuple[Category, str]]]:
  """
  Split each category's lyrics into k randomized, stratified folds for cross-validation.

  Args:
    lyrics_by_category (dict[str, list[str]]): A dictionary mapping categories to lists of lyrics.
    k (int): The number of folds to create. Default is 5.
    seed (int | None): An optional seed for the random number generator to ensure reproducibility. Default is None.

  Returns:
    list[list[tuple[Category, str]]]: A list of k folds, where each fold is a list of (category, lyric) tuples.
  """
  random_generator = random.Random(seed)
  # Initialize k empty folds
  folds: list[list[tuple[Category, str]]] = [[] for _ in range(k)]

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



def merge_training_sets(
  k_folds: list[list[tuple[Category, str]]],
  test_sets_index: int
) -> list[tuple[Category, str]]:
  """Merges the folds meant for training into a single list of training sets

  Args:
    k_folds (list[list[tuple[Category, str]]]): The split-up folds
    test_sets_index (int): The index of the fold used for testing

  Raises:
    ValueError: When the test sets index is invalid (negative or not a valid index)

  Returns:
    list[tuple[Category, str]]: The merged training sets
  """
  if test_sets_index < 0 or test_sets_index >= len(k_folds):
    raise ValueError("The test sets index is out of bounds")

  final_training_sets: list[tuple[Category, str]] = []
  
  for i, fold in enumerate(k_folds):
    if i == test_sets_index: continue
    final_training_sets.extend(fold)
  
  return final_training_sets



def show_metrics(metrics: dict[Metric, float]) -> None:
  """Displays the metrics of a run and their values

  Args:
    metrics (dict[Metric, float]): Metrics and their values
  """
  print()
  print("=========== Metrics ===========")
  for metric, value in metrics.items():
    print(f"{metric}: {value: 10}")
  print()
    
    
    
def show_statistics(all_metrics: defaultdict[Category, defaultdict[Metric, list[float]]]) -> None:
  """Calculates the means of each metric per category

  Args:
    all_metrics (defaultdict[Category, defaultdict[Metric, list[float]]]): Categories associated with metrics and list of values 
  """
  print()
  print("=======================================================")
  print(f"Metrics means")
  print("=======================================================")
  for category, metrics in all_metrics.items():
    print(f"{category}")
    
    for metric, values in metrics.items():
      metric_mean = sum(values) / len(values)
      print(f"{metric}: {metric_mean: 10}")
    print()
  print()
    
    
    
def five_cross_fold_validation(training_sets_path: str):
  """Performs a five fold cross validation"""
  # First load all of the lyrics in the training set
  try:
    training_sets: list[tuple[Category, str]] = load_lyrics(training_sets_path)
  except FileNotFoundError:
    print(f"File {training_sets_path} not found")
    return
  
  # Create dictionary grouping categories to training sets
  lyrics_by_category: dict[Category, list[str]] = map_training_sets_to_categories(training_sets) 
  
  # Obtain 5-fold split
  five_fold_split: list[list[tuple[Category, str]]] = k_fold_split(lyrics_by_category, k=5)
  
  # Index of the fold that will be used as testing set
  test_sets_index: int = 0 
    
  # category -> metric -> values from each fold
  all_metrics: defaultdict[
    Category,
    defaultdict[str, list[float]]
  ] = defaultdict(lambda: defaultdict(list))

  # In a loop of five iterations, train the Bayes Model with four folds and test with another
  for i in range(len(five_fold_split)):
    print("=======================================================")
    print(f"Iteration {i + 1} | Test sets {test_sets_index + 1}")
    print("=======================================================")
    
    current_test_sets: list[tuple[Category, str]] = five_fold_split[test_sets_index]
    current_training_sets: list[tuple[Category, str]] = merge_training_sets(five_fold_split, test_sets_index)
    current_naive_bayes = NaiveBayesModel(current_training_sets)
    
    # For each category in the model, calculate metrics by evaluating
    for category in current_naive_bayes.categories:
      current_metrics = evaluate_bayes_model(
        current_naive_bayes,
        current_test_sets,
        target_category=category)
    
      # Add each metric to the collection of all metrics
      for metric, value in current_metrics.items():
        all_metrics[category][metric].append(value)
    
      print(f"For {category}:")
      show_metrics(current_metrics)
    
    test_sets_index += 1

  # Final report of metric means
  show_statistics(all_metrics)



if __name__ == "__main__":
  from sys import argv, exit
  if len(argv) == 2:
    five_cross_fold_validation(argv[1])
  else:
    print(f"Usage: {argv[0]} {'{'}training-set{'}'}")
    exit()
