from dotenv import load_dotenv; load_dotenv()
import sys; sys.path.append(os.getenv('MP'))

import nn
import numpy as np

def prep_data(data, ratio, n_classes):
    # Split x, y
    x, y = data[:,:-1], data[:,-1]

    # OHE y
    y = np.eye(n_classes, dtype='int')[y]

    # Split train, test
    choice = np.random.choice(data.shape[0], size=int(ratio*data.shape[0]), replace=False)
    idx = np.zeros(data.shape[0], dtype=bool)
    idx[choice] = True

    x_train, x_val = x[~idx], x[idx]
    y_train, y_val = y[~idx], y[idx]

    return x_train, x_val, y_train, y_val

def main():
  data_path = r"C:\Users\rapha\Documents\datasets\tiny_stories\val_target2.npy"
  data = np.load(data_path)

  # Split data into x/y, train/test and OHE y
  x_train, x_val, y_train, y_val = prep_data(data, ratio=0.25, n_classes=301)

  # Batch data
  batch_size = 16
  x_train = nn.data.batch(x_train, batch_size).astype(int)
  y_train = nn.data.batch(y_train, batch_size).astype(int)
  x_val = nn.data.batch(x_val, batch_size).astype(int)
  y_val = nn.data.batch(y_val, batch_size).astype(int)

  np.save(r"C:\Users\rapha\Documents\datasets\tiny_stories\x_train.npy", x_train)
  np.save(r"C:\Users\rapha\Documents\datasets\tiny_stories\y_train.npy", y_train)
  np.save(r"C:\Users\rapha\Documents\datasets\tiny_stories\x_val.npy", x_val)
  np.save(r"C:\Users\rapha\Documents\datasets\tiny_stories\y_val.npy", y_val)