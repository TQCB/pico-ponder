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

  ##### MODEL DEFINITION #####

  # Model config
  # Embed size must be able to be split across heads
  output_size = 40
  embed_size = 32
  seq_len = 4
  n_heads = 8
  n_transformers = 2

  # Model definition
  model = nn.network.Network()

  def transformer_factory():
    return l.MetaLayer([
    l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads),
    l.LayerNormalisation(embed_size),
    l.Dense2D(input_dim=embed_size, output_dim=embed_size),
    l.Activation(nn.activations.Swish),
    l.Dense2D(input_dim=embed_size, output_dim=embed_size),
    l.Activation(nn.activations.Swish),
    l.LayerNormalisation(embed_size),
    ])

  model.add(l.PositionalEmbedding(seq_len=seq_len, output_dim=embed_size, vocab_size=output_size))
  for _ in range(n_transformers):
    model.add(transformer_factory())
  model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads, return_sequences=False))
  model.add(l.LayerNormalisation(embed_size))
  model.add(l.Dense1D(input_dim=embed_size, output_dim=output_size))
  model.add(l.Activation(nn.activations.Softmax))
 
  lr = nn.optimizers.LearningRateSchedule(1e-3)
  opt = nn.optimizers.AdamOptimizer

  model.build(nn.losses.cce, nn.losses.d_cce,
              nn.metrics.categorical_accuracy,
              optimizer=opt, 
              learning_rate_schedule=lr,)

  print(f"Batches/Epoch: {x_train.shape[0]}")
  print(f"Parameter count: {model.param_count}")
  
  lr_cb = nn.callbacks.PrintLRCallback()
  save_cb =  nn.callbacks.SaveOnProgressCallback(r'checkpoints')

  model.fit(x_train, y_train,
            x_val=x_val, y_val=y_val,
            validation=False,
            epochs=99_999,
            # callbacks=[lr_cb],
            batch_print_steps=100)

  with open('checkpoints/history.txt', 'w') as f:
    f.write(str(model.history))

if __name__ == '__main__':
  main()