from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import nn
import nn.layers as l
import numpy as np

def main():
  x_train = np.load(r'data/x_train.npy')
  y_train = np.load(r'data/y_train.npy')
  x_val = np.load(r'data/x_val.npy')
  y_val = np.load(r'data/y_val.npy')

  # Model config
  # Embed size must be able to be split across heads
  output_size = 300
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

  lr = nn.optimizers.LinearCycleSchedule(0.001, 0.1, 1000)
  save_cb =  nn.callbacks.SaveOnProgressCallback(r'data/model_checkpoints\first_run')

  model.build(nn.losses.cce, nn.losses.d_cce, nn.metrics.categorical_accuracy, lr)
  model.fit(x_train, y_train,
            x_val=x_val, y_val=y_val,
            epochs=3,
            callbacks=[save_cb],
            batch_print_steps=50)

if __name__ == '__main__':
  main()