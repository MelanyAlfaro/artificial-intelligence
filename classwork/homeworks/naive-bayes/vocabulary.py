def extract_vocabulary(bags: dict[str, dict[str, int]]) -> set:
  """
  Iterates through bags of words and merges the words into a single vocabulary.
  
  Args:
    bags (dict[str, dict]): Categories associated to the words and counts

  Returns:
    set: A set with all of the words in the corpus
  """

  # Use a set to represent vocabulary
  vocabulary: set = set()
  
  # Iterate through bags
  for bag in bags.values():
    # Unite (|) with current vocabulary so only unique words remain
    vocabulary |= bag.keys()
    
  return vocabulary
