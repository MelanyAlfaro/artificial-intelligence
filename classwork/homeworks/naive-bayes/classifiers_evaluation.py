"""
This module contains calculation functions applying the formulas for
accuracy, precision, recall, and f1. It also has a function for calculating all of the metrics.
"""

from common import Category
from lyrics_preprocessing import preprocess_lyrics
from naive_bayes_model import NaiveBayesModel

Metric = str

def calculate_accuracy(
  true_positives: int,
  false_positives: int,
  true_negatives: int,
  false_negatives: int
) -> float:
  """Returns accuracy (Proportion of predictions correct overall)"""
  try:
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
  except ZeroDivisionError:
    return 0.0

def calculate_precision(true_positives: int, false_positives) -> float:
  """Returns precision (Proportion of the predicted positives that were actually positive)"""
  try:
    return true_positives / (true_positives + false_positives)
  except ZeroDivisionError:
    return 0.0

def calculate_recall(true_positives: int, false_negatives: int) -> float:
  """Returns recall (Proportion of real positives that were correctly identified)"""
  try:   
    return true_positives / (true_positives + false_negatives)
  except ZeroDivisionError:
    return 0.0

def calculate_f1(precision: float, recall: float) -> float:
  """Calculates f1 (Harmonic mean of precision and recall)"""
  try:
    return 2 * (precision * recall) / (precision + recall)
  except ZeroDivisionError:
    return 0.0

def evaluate_bayes_model(
  naive_bayes_model: NaiveBayesModel,
  test_sets: list[tuple[Category, str]],
  target_category: Category
) -> dict[Metric, float]:
  """Evaluates bayes model by calculating accuracy, precision, recall, and f1 based on a built corpus
  and test sets

  Args:
    naive_bayes_model (NaiveBayesModel): The Bayes Model to evaluate, which would be trained upon creation and ready for classification
    test_sets (list[tuple[Category, str]]): List of category, lyric pairs
    target_category (Category): The category to evaluate the model for

  Returns:
    dict[Metric, float]: A dictionary with each metric
  """
  # Safeguard to avoid unnecessary evaluation
  if target_category not in naive_bayes_model.categories:
    raise ValueError("The target category is not registered in the model")

  true_positives: int = 0
  false_positives: int = 0
  false_negatives: int = 0
  true_negatives: int = 0

  for category, test_set in test_sets:
    clean_test_set = preprocess_lyrics(test_set)
    classified_cat = naive_bayes_model.classify(clean_test_set)
    
    # True positive
    if category == target_category and classified_cat == target_category:
      true_positives += 1
    # False negative 
    elif category == target_category and classified_cat != target_category:
      false_negatives += 1
    # False positive
    elif category != target_category and classified_cat == target_category:
      false_positives += 1
    # True negative
    elif category != target_category and classified_cat != target_category:
      true_negatives += 1

  metrics = {}
  metrics["accuracy"] = calculate_accuracy(true_positives, false_positives, true_negatives, false_negatives)
  metrics["precision"] = calculate_precision(true_positives, false_positives)
  metrics["recall"] = calculate_recall(true_positives, false_negatives)
  metrics["f1"] = calculate_f1(metrics["precision"], metrics["recall"])
  
  return metrics
