import numpy as np

import sys
sys.path.append(r"C:\Users\rapha\My Drive\Work\data_science\projects\scratch_nn")
import nn
import nn.layers as l

def main():
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

  data_path = r"C:\Users\rapha\Documents\datasets\tiny_stories\val_target2.npy"
  data = np.load(data_path)

  # Split data into x/y, train/test and OHE y
  x_train, x_val, y_train, y_val = prep_data(data, ratio=0.25, n_classes=301)

  # Batch data
  batch_size = 16
  x_train = nn.data.batch(x_train, batch_size)
  y_train = nn.data.batch(y_train, batch_size)
  x_val = nn.data.batch(x_val, batch_size)
  y_val = nn.data.batch(y_val, batch_size)

  # Model config
  # Embed size must be able to be split across heads
  output_size = 301
  embed_size = 16
  seq_len = 64
  n_heads = 4

  n_transformers = 3

  # Model definition
  model = nn.network.Network()

  transformer = l.MetaLayer([
    l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads),
    l.LayerNormalisation(embed_size),
    l.Dense2D(input_dim=embed_size, output_dim=embed_size),
    l.Activation(nn.activations.Swish),
    l.LayerNormalisation(embed_size)
  ])

  model.add(l.PositionalEmbedding(seq_len=seq_len, output_dim=embed_size, vocab_size=output_size))
  for i in range(n_transformers):
    model.add(transformer)
  model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads, return_sequences=False))
  model.add(l.Dense1D(input_dim=embed_size, output_dim=output_size))
  model.add(l.Activation(nn.activations.Softmax))

  # lr = nn.optimizers.LinearCycleSchedule(0.01, 1.0, 1000)
  lr = nn.optimizers.LearningRateSchedule(0.1)
  save_cb =  nn.callbacks.SaveOnProgressCallback(r"C:\Users\rapha\Documents\datasets\tiny_stories\model_checkpoints\first_run")

  model.build(nn.losses.cce, nn.losses.d_cce, nn.metrics.categorical_accuracy, lr)
  model.fit(x_train, y_train,
            x_val=x_val, y_val=y_val,
            epochs=3,
            callbacks=[save_cb])

if __name__ == '__main__':
  main()