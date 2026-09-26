"""
  This module provides functions that may be used in different parts of the naive bayes homework.
"""
import csv


def load_lyrics(file_path: str) -> list:
    """
    Load lyrics from CSV file 

    The CSV is expected to have two columbs : CATEGORIA and the lyrics fragment. 
    Lyrics may span multiple lines and contain blank lines.

    Args:
        file_path (str): The path to the CSV file containing the lyrics.
    
    Returns:
        list: A list of (category, lyrics) tuples, one per row in the CSV file.
    """
    
    lyrics = []

    with open(file_path, mode="r", encoding= "utf-8", newline= "") as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader: 
        # Skip the row immediately if every field is blank/whitespace.
        if not any((value or "").strip() for value in row.values()):
          continue

        category = row["CATEGORIA"].strip()
        # Grab whichever colum isn't CATEGORIA, so this still works if the lyrics column is renamed
        lyrics_key = next (key for key in row if key != "CATEGORIA")
        lyrics_text = row[lyrics_key]

        lyrics.append((category, lyrics_text))
    
    print (f"Loaded {len(lyrics)} lyrics from {file_path}")
    return lyrics