from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import femto_flow as ff
import numpy as np
import pandas as pd

def encode(df, column):
  df = df.copy()
  
  cats = df[column].unique()
  encoding = {}
  for i, cat in enumerate(cats):
    encoding[cat] = i

  df[column] = df[column].map(encoding)
  return df

def main():
  vocab_size = 1000
  verbosity = 2

  df = pd.read_csv(r"demos/classification_demo/data/tweet_emotions.csv")
  df = df.drop('tweet_id', axis=1)
  df = df.loc[:1000]

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
  df['vectors'] = df['tokens'].apply(lambda x: vectorizer.transform(x))

  print(f"Vocabulary: {vectorizer.vocabulary}\n") if verbosity > 1 else None
  print(f"{df.loc[:10, 'vectors']}\n") if verbosity > 1 else None

  # print(f"Unknown Token Frequency: {len([s for s in sequences[0] if s == 0]) / len(sequences[0]) * 100:.1f}%\n") if verbosity > 0 else None
  
  x_train = np.array(df['vectors'])
  print(x_train)

if __name__ == '__main__':
  main()