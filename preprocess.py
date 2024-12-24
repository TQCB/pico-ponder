import os
import sys
import numpy as np

from dotenv import load_dotenv; load_dotenv()
import sys; sys.path.append(os.getenv('MP'))

import nn

def sgen(filepath, sep='<|endoftext|>'):
  """
  A generator that yields sentences from a text file.

  Args:
    filepath: Path to text file
    sep: Word that separates sentences
  
  Yields:
    str: Sentences found in the file
  """
  with open(filepath, 'r', encoding='utf-8') as f:
    current_sentence = ""
    for line in f:
      for word in line.split(): # split by spaces
        if word == sep:
          yield current_sentence.strip()
          current_sentence = ""
        else:
          current_sentence += " " + word
    # yield the last sentence, if not ending with sep.
    if current_sentence.strip():
      yield current_sentence.strip()

def targets_from_sequence(sequence, context_size):
  """
  Constructs n times X, y training data from a sequence. X being our
  context_size input,  y the target, and n the amount of targets created from a
  sequence (should be equal to len(sequence) - 1, since every word can be a
  target except the first)

  X is context_size words used to predict context_size + 1 word y

  Args:
    sequence (array): array of ints, vectorised tokens
    context_size (int): length of X for each X, y pair returned
  
  Returns:
    list[(X,y)]: n long list containing tuples of X, y pairs
  """
  target_amount = len(sequence) - 1

  # Ensure input in an np.ndarray
  sequence = np.array(sequence)

  # Result array is a matrix for (n, (X,y))
  # n = target_amount, X = context_size, y = 1
  result = np.zeros((target_amount, context_size+1))


  for i in range(target_amount):
    # Get index of context start, unless its negative
    idx = i - context_size + 1
    context_start = max(0, idx)

    # Get sequence and target, ensuring they are arrays
    context = sequence[context_start:i+1]
    target = np.array(sequence[i+1], ndmin=1)

    # If our context is shorter than context size, we will pad context
    if context_start == 0:
      context = np.pad(context, (0, np.abs(idx)), mode='constant')

    result[i,:] = np.concatenate([context, target])
    
  return result

def targets_from_sequences(sequences, context_size):
  """Calls targets_from_sequences() on a list of sequences"""
  result = []
  for sequence in sequences:
    result.append(targets_from_sequence(sequence, context_size))
  return np.vstack(result)

def main():
  train_path = r"C:\Users\rapha\Documents\datasets\tiny_stories\TinyStoriesV2-GPT4-train.txt"
  val_path = r"C:\Users\rapha\Documents\datasets\tiny_stories\TinyStoriesV2-GPT4-valid.txt"

  # Initialise tokeniser and vectoriser
  tokenizer = nn.text.BytePairTokenizer(300)
  vectorizer = nn.text.Vectorizer(300)

  # Fit tokeniser
  tokenizer.fit(sgen(val_path))

  # Get train and val tokens
  # train_tokens = tokenizer.transform(sgen(train_path))
  val_tokens = tokenizer.transform(sgen(val_path))
  v_fit_tokens = tokenizer.transform(sgen(val_path))

  # Fit vectoriser
  vectorizer.fit(v_fit_tokens)

  # Get train and val sequences
  # train_sequences = vectorizer.transform(train_tokens)
  val_sequences = vectorizer.transform(val_tokens)

  # train_data = targets_from_sequences(train_sequences, 64)
  val_data = targets_from_sequences(val_sequences, 64)
  val_data = val_data.astype(int)

  val_save_path = r"C:\Users\rapha\Documents\datasets\tiny_stories\val_target2.npy"
  np.save(val_save_path, val_data)

if __name__ == '__main__':
  main()