import os
import sys
import numpy as np

from dotenv import load_dotenv; load_dotenv()
import sys; sys.path.append(os.getenv('MP'))

import nn

def load_data(path):
  with open(path, 'r') as f:
    data = f.read()
  return data

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

def one_hot_encode(x, n_classes=None, dtype='int'):
  if n_classes is None:
    n_classes = np.max(x) # max class number found in array
  return np.eye(n_classes, dtype=dtype)[x]

def train_split(data, ratio, n_classes):
    # Split x, y
    x, y = data[:,:-1], data[:,-1]

    # Split train, test
    choice = np.random.choice(data.shape[0], size=int(ratio*data.shape[0]), replace=False)
    idx = np.zeros(data.shape[0], dtype=bool)
    idx[choice] = True

    x_train, x_val = x[~idx], x[idx]
    y_train, y_val = y[~idx], y[idx]

    return x_train, x_val, y_train, y_val

def main():
  batch_size = 4
  vocab_size = 300
  data_path = r"data/pico_stories.csv"

  data = load_data(data_path)

  # Initialise tokeniser and vectoriser
  tokenizer = nn.text.BytePairTokenizer(vocab_size)
  vectorizer = nn.text.Vectorizer(vocab_size)

  # Fit tokeniser
  tokenizer.fit(data)
  tokens = tokenizer.transform(data)

  # Fit vectoriser
  vectorizer.fit(tokens)
  sequences = vectorizer.transform(tokens)

  # Get target sequences
  data = targets_from_sequences(sequences, 64).astype(int)

  # Split and encode
  x_train, x_val, y_train, y_val = train_split(data, 0.2)
  y_train, y_val = one_hot_encode(y_train, n_classes=vocab_size), one_hot_encode(y_val, n_classes=vocab_size)

  # Batch data
  from nn.data import batch
  x_train = batch(x_train, batch_size)
  y_train = batch(y_train, batch_size)
  x_val = batch(x_val, batch_size)
  y_val = batch(y_val, batch_size)

  # Save data
  np.save(x_train, r"data/x_train.npy")
  np.save(y_train, r"data/y_train.npy")
  np.save(x_val, r"data/x_val.npy")
  np.save(y_val, r"data/y_val.npy")

if __name__ == '__main__':
  main()