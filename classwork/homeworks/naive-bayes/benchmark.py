from naive_bayes_model import NaiveBayesModel
from lyrics_preprocessing import load_lyrics
from common import Category
from classifiers_evaluation import Metric, evaluate_bayes_model
from collections import defaultdict
import statistics as stat


def show_results(metrics: dict[Category, dict[Metric, float]]) -> None:
  all_metrics: dict[Metric, list[float]] = defaultdict(list[float])
  print()
  print(f"{"=" * (30 + len("Results"))}")
  print(f"{"Results":>{30 // 2 + len("Results")}}")
  print(f"{"=" * (30 + len("Results"))}")
  print("Values per category:")
  col_widths = [12, 12, 12]
  col_precicions = [10]
  print(f"{"Category":{col_widths[0]}}{"Field":{col_widths[1]}}{"Value":{col_widths[2]}}")
  for cat, cat_metric in metrics.items():
    for metric, value in cat_metric.items():
      all_metrics[metric].append(value)
      print(f"{cat:{col_widths[0]}}{metric:{col_widths[1]}}{value:<0{col_widths[2]}.{col_precicions[0]}}")

  mean_metrics: dict[Metric, float] = dict[Metric, float]()
  for metric, values in all_metrics.items():
    mean_metrics[metric] = stat.mean(values)
  print()
  print("Means:")
  col_widths = [12, 12]
  col_precicions = [10]
  print(f"{"Field":{col_widths[0]}}{"Mean":{col_widths[1]}}")
  for metric, mean_value in mean_metrics.items():
    print(f"{metric:{col_widths[0]}}{mean_value:<0{col_widths[1]}.{col_precicions[0]}}")


def benchmark(training: str, testing: str) -> None:
  training_lyrics = load_lyrics(training)
  testing_lyrics = load_lyrics(testing)
  model = NaiveBayesModel(training_sets=training_lyrics)
  metrics: dict[Category, dict[Metric, float]] = defaultdict(dict)
  for cat in model.categories:
    metrics[cat] = evaluate_bayes_model(
      naive_bayes_model=model,
      test_sets=testing_lyrics,
      target_category=cat)

  show_results(metrics)


if __name__ == "__main__":
  from sys import argv
  if len(argv) == 3:
    benchmark(argv[1], argv[2])
  else:
    print(f"Usage: {argv[0]} {'{'}training-set{'}'} {'{'}testing-set{'}'}")
    exit(0)
