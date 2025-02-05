from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import femto_flow as ff
import numpy as np
import pandas as pd

def pad_or_slice(arr, target):
    if arr.shape[0] < target:
      arr = np.pad(arr, (0, target - arr.shape[0]))
      return arr
    
    elif arr.shape[0] > target:
      arr = arr[:target]
      return arr
    
    else:
      return arr

def encode(df, column):
  df = df.copy()
  
  cats = df[column].unique()
  encoding = {}
  for i, cat in enumerate(cats):
    encoding[cat] = i

  df[column] = df[column].map(encoding)
  return df

def train_test_split(x, y, ratio):
    # Split train, test
    choice = np.random.choice(x.shape[0], size=int(ratio*x.shape[0]), replace=False)
    idx = np.zeros(x.shape[0], dtype=bool)
    idx[choice] = True

    x_train, x_val = x[~idx], x[idx]
    y_train, y_val = y[~idx], y[idx]

    return x_train, x_val, y_train, y_val

def main():
  vocab_size = 1000
  seq_len = 32
  verbosity = 1
  batch_size = 32

  df = pd.read_csv(r"demos/classification_demo/data/tweet_emotions.csv")
  df = df.drop('tweet_id', axis=1)

  ##### Encode sentiment column #####
  df = encode(df, 'sentiment')
  corpus = df['content'].values
  corpus = [doc.lower() for doc in corpus]
  
  print(f"Corpus length: {len(corpus)}") if verbosity > 1 else None

  ##### Tokenize and vectorize text content #####
  tokenizer = ff.text.BytePairTokenizer(vocab_size)
  vectorizer = ff.text.Vectorizer(vocab_size)

  tokenizer.fit(corpus)
  df['tokens'] = df['content'].apply(lambda x: tokenizer.transform([x]))
  
  print(f"Tokens: {df.loc[:10, 'tokens']}\n") if verbosity > 1 else None
  print(f"Learnt merges: {tokenizer.merges}\n") if verbosity > 1 else None

  vectorizer.fit([s[0] for s in df.loc[:10, 'tokens'].values.tolist()])
  df['vectors'] = df['tokens'].apply(lambda x: np.array(vectorizer.transform(x)[0]))

  print(f"Vocabulary: {vectorizer.vocabulary}\n") if verbosity > 1 else None
  print(f"{df.loc[:10, 'vectors']}\n") if verbosity > 1 else None

  # print(f"Unknown Token Frequency: {len([s for s in sequences[0] if s == 0]) / len(sequences[0]) * 100:.1f}%\n") if verbosity > 0 else None

  df['vectors'] = df['vectors'].apply(lambda x: pad_or_slice(x, seq_len))

  # Add all elements of vector column to array
  x = np.empty((df.shape[0], seq_len))
  for i in range(df.shape[0]):
    x[i] = df.loc[i, 'vectors']

  # Creating y from one hot encoded sentiment information
  n_cat = df['sentiment'].nunique()
  y = np.empty((df.shape[0], n_cat))
  for i in range(df.shape[0]):
    oh_vec = np.zeros(n_cat)
    oh_vec[df.loc[i, 'sentiment']] = 1
    y[i] = oh_vec

  # Split data
  x_train, x_val, y_train, y_val = train_test_split(x, y, 0.2)

  # Batch data
  x_train = ff.data.batch(x_train, batch_size)
  y_train = ff.data.batch(y_train, batch_size)
  x_val = ff.data.batch(x_val, batch_size)
  y_val = ff.data.batch(y_val, batch_size)

  # Save data
  np.save(r"demos/classification_demo/data/x_train.npy", x_train)
  np.save(r"demos/classification_demo/data/y_train.npy", y_train)
  np.save(r"demos/classification_demo/data/x_val.npy", x_val)
  np.save(r"demos/classification_demo/data/y_val.npy", y_val)

if __name__ == '__main__':
  main()