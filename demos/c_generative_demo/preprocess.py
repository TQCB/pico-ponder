from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import numpy as np
import pickle
import femto_flow as ff

from femto_flow.data import load_data, targets_from_sequences, one_hot_encode, train_test_split

def main():
  # If you want to see info on preprocessing
  # Accepts verbosities of 0, 1, 2 for No output, Minimal output, Full output
  verbosity = 1
  
  # Limit input data in case I want to test
  char_limit = 1_000
  
  # Set preprocessing paramaters
  batch_size = 8
  seq_len = 8
  vocab_size = 64
  sep = '<|endoftext|>' # sometimes necessary if separating training inputs instead of training continously
  data_path = r"demos/c_generative_demo/data/drseuss.txt"

  data = load_data(data_path, encoding='latin1')
  if char_limit > 0:
    data = data[:char_limit]
  data = data.lower()
  data = data.replace("\n", " ")
  data = [list(data)]

  # Initialize vectoriser
  vectorizer = ff.text.Vectorizer(vocab_size)

  # Fit vectoriser
  vectorizer.fit(data)
  sequences = vectorizer.transform(data)
  print(vectorizer.vocabulary)

  # Verbosity prints
  if verbosity > 0:
    if verbosity > 1:
      # Print tokenizer, vocab and output info for eventual debugging
      print(f"Sequences: {sequences}\n")
      print(f"Vectorizer vocab: {vectorizer.vocabulary}\n")
    # Useful to see how much of your vocab is left out
    print(f"Unknown Token Frequency: {len([s for s in sequences[0] if s == 0]) / len(sequences[0]) * 100:.1f}%\n")

  with open(r'demos/c_generative_demo/checkpoints/word_processor.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

  # Get target sequences
  data = targets_from_sequences(sequences, seq_len).astype(int)

  # Split and encode
  x_train, x_val, y_train, y_val = train_test_split(data, 0.2)
  y_train = one_hot_encode(y_train, n_classes=vocab_size)
  y_val = one_hot_encode(y_val, n_classes=vocab_size)

  # Batch data
  x_train = ff.data.batch(x_train, batch_size)
  y_train = ff.data.batch(y_train, batch_size)
  x_val = ff.data.batch(x_val, batch_size)
  y_val = ff.data.batch(y_val, batch_size)

  # Save data
  np.save(r"demos/c_generative_demo/data/x_train.npy", x_train)
  np.save(r"demos/c_generative_demo/data/y_train.npy", y_train)
  np.save(r"demos/c_generative_demo/data/x_val.npy", x_val)
  np.save(r"demos/c_generative_demo/data/y_val.npy", y_val)

if __name__ == '__main__':
  main()