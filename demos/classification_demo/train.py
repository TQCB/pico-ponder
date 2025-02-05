from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import femto_flow as ff
import femto_flow.layers as l
import numpy as np

def main():
  x_train = np.load(r'demos/classification_demo/data/x_train.npy').astype(int)
  y_train = np.load(r'demos/classification_demo/data/y_train.npy').astype(int)
  x_val = np.load(r'demos/classification_demo/data/x_val.npy').astype(int)
  y_val = np.load(r'demos/classification_demo/data/y_val.npy').astype(int)

  ##### MODEL DEFINITION #####

  # Model config
  vocab_size = 1000
  output_size = 13
  embed_size = 32
  seq_len = 32
  n_heads = 1
  n_transformers = 3
  dropout_rate = 0.0
  
  # Embed size must be able to be split across heads
  assert embed_size % n_heads == 0, "Embed size cannot be split across attention heads."
  
  # Training config
  epochs = 10
  lr = 1e-3

  # Model definition
  model = ff.network.Network()

  def create_transformer():
    return l.MetaLayer([
    l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads),
    l.LayerNormalisation(embed_size),
    l.Dense2D(input_dim=embed_size, output_dim=embed_size),
    l.InvertedDropout(dropout_rate),
    l.Activation(ff.activations.Swish),
    l.Dense2D(input_dim=embed_size, output_dim=embed_size),
    l.InvertedDropout(dropout_rate),
    l.Activation(ff.activations.Swish),
    l.LayerNormalisation(embed_size),
    ])

  model.add(l.PositionalEmbedding(seq_len=seq_len, output_dim=embed_size, vocab_size=vocab_size))
  for _ in range(n_transformers):
    model.add(create_transformer())
  model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads, return_sequences=False))
  model.add(l.LayerNormalisation(embed_size))
  model.add(l.Dense1D(input_dim=embed_size, output_dim=output_size))
  model.add(l.Activation(ff.activations.Softmax))

  lrs = ff.optimizers.LearningRateSchedule(lr)

  opt = ff.optimizers.AdamOptimizer
  
  model.build(ff.losses.cce, ff.losses.d_cce,
              ff.metrics.categorical_accuracy,
              optimizer=opt, 
              learning_rate_schedule=lrs,)

  print(f"Batches per epoch: {x_train.shape[0]}")
  print(f"Parameter count:   {model.param_count:,.0f}")
  
  save_cb =  ff.callbacks.SaveOnProgressCallback(r'demos/classification_demo/checkpoints')

  model.fit(x_train, y_train,
            x_val=x_val, y_val=y_val,
            validation=True,
            epochs=epochs,
            # callbacks=[save_cb],
            batch_print_steps=10)

  # with open('demos/classification_demo/checkpoints/history.txt', 'w') as f:
  #   f.write(str(model.history))

if __name__ == '__main__':
  main()