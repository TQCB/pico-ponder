from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import femto_flow as ff
import femto_flow.layers as l
import numpy as np

def main():
  x_train = np.load(r'data/x_train.npy')
  y_train = np.load(r'data/y_train.npy')
  x_val = np.load(r'data/x_val.npy')
  y_val = np.load(r'data/y_val.npy')

  ##### MODEL DEFINITION #####

  # Model config
  # Embed size must be able to be split across heads
  output_size = 256
  embed_size = 64
  seq_len = 2
  n_heads = 8
  n_transformers = 2

  # Model definition
  model = ff.network.Network()

  def transformer_factory():
    return l.MetaLayer([
    l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads),
    l.LayerNormalisation(embed_size),
    l.Dense2D(input_dim=embed_size, output_dim=embed_size),
    l.Activation(ff.activations.Swish),
    l.Dense2D(input_dim=embed_size, output_dim=embed_size),
    l.InvertedDropout(0.2),
    l.Activation(ff.activations.Swish),
    l.LayerNormalisation(embed_size),
    ])

  model.add(l.PositionalEmbedding(seq_len=seq_len, output_dim=embed_size, vocab_size=output_size))
  for _ in range(n_transformers):
    model.add(transformer_factory())
  model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads, return_sequences=False))
  model.add(l.LayerNormalisation(embed_size))
  model.add(l.Dense1D(input_dim=embed_size, output_dim=output_size))

  model.add(l.Activation(ff.activations.Softmax))
 
  lr = ff.optimizers.LearningRateSchedule(1e-3)
  lr = ff.optimizers.ExponentialDecaySchedule(1e-2, 100, 0.8, min=1e-4)
  opt = ff.optimizers.AdamOptimizer

  model.build(ff.losses.cce, ff.losses.d_cce,
              ff.metrics.categorical_accuracy,
              optimizer=opt, 
              learning_rate_schedule=lr,)

  print(f"Batches/Epoch: {x_train.shape[0]}")
  print(f"Parameter count: {model.param_count}")
  
  lr_cb = ff.callbacks.PrintLRCallback()
  save_cb =  ff.callbacks.SaveOnProgressCallback(r'checkpoints')

  model.fit(x_train, y_train,
            x_val=x_val, y_val=y_val,
            validation=True,
            epochs=30,
            # callbacks=[lr_cb],
            batch_print_steps=20)

  with open('checkpoints/history.txt', 'w') as f:
    f.write(str(model.history))

if __name__ == '__main__':
  main()