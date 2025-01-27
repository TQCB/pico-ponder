from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import numpy as np
import pickle
import femto_flow as ff

def load_data(path):
  if path == "data/drseuss.txt":
    with open(path, 'r') as f:
      data = f.read()
  else:
    with open(path, 'r', encoding='utf-8') as f:
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

def train_split(data, ratio):
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
  # If you want to see info on preprocessing
  # Accepts verbosities of 0, 1, 2 for No output, Minimal output, Full output
  verbosity = 1
  
  # Limit input data in case I want to test
  char_limit = 0
  
  # Set preprocessing paramaters
  batch_size = 8
  seq_len = 16
  vocab_size = 64
  sep = '<|endoftext|>' # sometimes necessary if separating training inputs instead of training continously
  data_path = r"data/drseuss.txt"

  data = load_data(data_path)
  if char_limit > 0:
    data = data[:char_limit]
  data = data.lower()
  data = data.replace("\n", " ")
  data = [list(data)]

  # Initialise tokeniser and vectoriser
  vectorizer = ff.text.Vectorizer(vocab_size)

  # Fit vectoriser
  vectorizer.fit(data)
  sequences = vectorizer.transform(data)

  # Verbosity prints
  if verbosity > 0:
    if verbosity > 1:
      # Print tokenizer, vocab and output info for eventual debugging
      print(f"Sequences: {sequences}\n")
      print(f"Vectorizer vocab: {vectorizer.vocabulary}\n")
    # Useful to see how much of your vocab is left out
    print(f"Unknown Token Frequency: {len([s for s in sequences[0] if s == 0]) / len(sequences[0]) * 100:.1f}%\n")

  with open(r'checkpoints/char_gen/word_processor.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

  # Get target sequences
  data = targets_from_sequences(sequences, seq_len).astype(int)

  # Split and encode
  x_train, x_val, y_train, y_val = train_split(data, 0.2)
  y_train = one_hot_encode(y_train, n_classes=vocab_size)
  y_val = one_hot_encode(y_val, n_classes=vocab_size)

  # Batch data
  x_train = ff.data.batch(x_train, batch_size)
  y_train = ff.data.batch(y_train, batch_size)
  x_val = ff.data.batch(x_val, batch_size)
  y_val = ff.data.batch(y_val, batch_size)

  # Save data
  np.save(r"data/x_train.npy", x_train)
  np.save(r"data/y_train.npy", y_train)
  np.save(r"data/x_val.npy", x_val)
  np.save(r"data/y_val.npy", y_val)

if __name__ == '__main__':
  main()