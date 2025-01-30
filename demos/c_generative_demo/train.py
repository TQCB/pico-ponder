from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import femto_flow as ff
import femto_flow.layers as l
import numpy as np

def main():
  x_train = np.load(r'demos/c_generative_demo/data/x_train.npy')
  y_train = np.load(r'demos/c_generative_demo/data/y_train.npy')
  x_val = np.load(r'demos/c_generative_demo/data/x_val.npy')
  y_val = np.load(r'demos/c_generative_demo/data/y_val.npy')

  ##### MODEL DEFINITION #####

  # Model config
  output_size = 64
  embed_size = 128
  seq_len = 8
  n_heads = 4
  n_transformers = 3
  dropout_rate = 0.0
  
  # Embed size must be able to be split across heads
  assert embed_size % n_heads == 0, "Embed size cannot be split across attention heads."
  
  # Training config
  epochs = 50

  decay = False
  max_lr = 1e-4
  min_lr = 1e-4
  decay_rate = (min_lr/max_lr) ** (1/epochs) # decay rate to achieve min_lr from max_lr over n epochs

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

  model.add(l.PositionalEmbedding(seq_len=seq_len, output_dim=embed_size, vocab_size=output_size))
  for _ in range(n_transformers):
    model.add(create_transformer())
  model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads, return_sequences=False))
  model.add(l.LayerNormalisation(embed_size))
  model.add(l.Dense1D(input_dim=embed_size, output_dim=output_size))
  model.add(l.Activation(ff.activations.Softmax))

  if decay:
    lr = ff.optimizers.ExponentialDecaySchedule(initial_lr=max_lr,
                                                decay_steps=x_train.shape[0],
                                                decay_rate=decay_rate,
                                                min=min_lr)
    print(f"Epoch LR decay rate: {decay_rate:.3f}")

  else:
    lr = ff.optimizers.LearningRateSchedule(1e-3)

  opt = ff.optimizers.AdamOptimizer
  
  model.build(ff.losses.cce, ff.losses.d_cce,
              ff.metrics.categorical_accuracy,
              optimizer=opt, 
              learning_rate_schedule=lr,)

  print(f"Batches per epoch: {x_train.shape[0]}")
  print(f"Parameter count:   {model.param_count:,.0f}")
  
  lr_cb = ff.callbacks.PrintLRCallback()
  save_cb =  ff.callbacks.SaveOnProgressCallback(r'demos/c_generative_demo/checkpoints')

  model.fit(x_train, y_train,
            x_val=x_val, y_val=y_val,
            validation=True,
            epochs=epochs,
            callbacks=[save_cb],
            batch_print_steps=10)

  with open('demos/c_generative_demo/checkpoints/history.txt', 'w') as f:
    f.write(str(model.history))

if __name__ == '__main__':
  main()