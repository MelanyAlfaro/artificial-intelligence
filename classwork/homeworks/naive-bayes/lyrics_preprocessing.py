
"""
This module provides functionality to load lyrics from a CSV file
and return a dictonary of bag of words representation by category 
with the text normalized to lowercase, removing accented 
characters and punctuation, and keeping only letters 
from 'a' to 'z' in the spanish alphabet.
"""

import json
import re
from collections import Counter

from common import load_lyrics


def remove_accents(text: str) -> str:
    """
    Remove accented vowels from the text

    Args:
        text (str): The text from which to remove accented vowels.

    Returns:
        str: The text with accented vowels replaced by their unaccented counterparts.
    """
    replacements = str.maketrans("áéíóúü", "aeiouu")
    return text.translate(replacements)
    

def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from the text.

    Args:
        text (str): The text from which to remove punctuation.

    Returns:
        str: The text with punctuation removed.
    """
    return re.sub(r"[^a-zñ\s]", "", text)


def preprocess_lyrics(lyrics: str) -> str:
    """
    Preprocess the lyrics by normalizing to lowercase, removing accented characters,
    punctuation, and keeping only letters from 'a' to 'z'.

    Args:
        lyrics (str): The lyrics to preprocess.

    Returns:
        str: The preprocessed lyrics.
    """

    text = lyrics.lower()
    # Remove accented vowels from the text
    text = remove_accents(text)
    # Remove punctuation and keep only letters from 'a' to 'z'
    text = remove_punctuation(text)
    # Remove extra whitespace and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()  
    return text

  
def create_bag_of_words(lyrics: list) -> dict:
    """
    Create a bag of words representation from the preprocessed lyrics.

    Args: 
        lyrics (list): A list of (category, lyrics) tuples, as returned by load_lyrics.

    Returns:
      A dictionary where keys are the categories and values are of the form {word: count} representing the bag of words for that category.
    """
    bags = {}


    for category, text_lyrics in lyrics:
      # We process csv lyrics cells one by one 
      cleaned_lyrics = preprocess_lyrics(text_lyrics)
      words = cleaned_lyrics.split()
      bags.setdefault(category, Counter()).update(words)

    return bags

def save_bag_of_words(bags: dict, output_path : str) -> None:
    """
    Save the bag of words representation to a file. Its usage 
    is optional, but it can be useful to save the bag of words for later use.

    Returns:
        None
    """
    serialized_bags = {
       category: dict(counter.most_common())
       for category, counter in bags.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized_bags, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    lyrics = load_lyrics("data/training-lyrics.csv")
    bags = create_bag_of_words(lyrics)
    save_bag_of_words(bags, "data/bag_of_words.json")