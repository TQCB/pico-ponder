from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import nn
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

  df = pd.read_csv(r"C:\Users\rapha\Documents\datasets\tweet_emotions\tweet_emotions.csv")
  df = df.drop('tweet_id', axis=1)

  ##### Encode sentiment column #####
  df = encode(df, 'sentiment')

  ##### Tokenize and vectorize text content #####
  tokenizer = nn.text.BytePairTokenizer(vocab_size)
  vectorizer = nn.text.Vectorizer(vocab_size)

  tokenizer.fit(df)
  tokens = tokenizer.transform(df)
  print(tokens)

  vectorizer.fit(tokens)
  sequences = vectorizer.transform(tokens)

  print(sequences)

if __name__ == '__main__':
  main()