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
  output_size = 64
  embed_size = 64
  seq_len = 4
  n_heads = 8
  n_transformers = 2

  # Model definition
  model = nn.network.Network()

  # THESE DON'T WORK BECAUSE THEY SHARE WEIGHTS!!!!!
  # Need some sort of factory to generate these independently

  # ffn = l.MetaLayer([
  #   l.Dense2D(input_dim=embed_size, output_dim=embed_size),
  #   l.Activation(nn.activations.Swish),
  #   l.Dense2D(input_dim=embed_size, output_dim=embed_size),
  #   l.Activation(nn.activations.Swish),
  # ], clip=1)

  # transformer = l.MetaLayer([
  #   l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads),
  #   l.LayerNormalisation(embed_size),
  #   l.Dense2D(input_dim=embed_size, output_dim=embed_size),
  #   l.Activation(nn.activations.Swish),
  #   l.Dense2D(input_dim=embed_size, output_dim=embed_size),
  #   l.Activation(nn.activations.Swish),
  #   l.LayerNormalisation(embed_size),
  # ], clip=1)

  model.add(l.PositionalEmbedding(seq_len=seq_len, output_dim=embed_size, vocab_size=output_size))
  
  # transformer
  model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads))
  model.add(l.LayerNormalisation(embed_size))
  model.add(l.Dense2D(input_dim=embed_size, output_dim=embed_size))
  model.add(l.Activation(nn.activations.Swish))
  model.add(l.Dense2D(input_dim=embed_size, output_dim=embed_size))
  model.add(l.Activation(nn.activations.Swish))
  model.add(l.LayerNormalisation(embed_size))

  model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads, return_sequences=False))
  model.add(l.LayerNormalisation(embed_size))
  model.add(l.Dense1D(input_dim=embed_size, output_dim=output_size))
  model.add(l.Activation(nn.activations.Softmax))
 
  lr = nn.optimizers.LearningRateSchedule(1e-5)
  opt = nn.optimizers.AdamOptimizer
  # lr = nn.optimizers.ExponentialDecaySchedule(initial_lr=1e-1, decay_steps=100, decay_rate=0.96, min=1e-3)
  save_cb =  nn.callbacks.SaveOnProgressCallback(r'checkpoints')

  model.build(nn.losses.cce, nn.losses.d_cce,
              nn.metrics.categorical_accuracy,
              optimizer=opt, 
              learning_rate_schedule=lr,)

  print(f"Parameter count: {model.param_count}")

  model.fit(x_train, y_train,
            # x_val=x_val, y_val=y_val,
            validation=False,
            epochs=100,
            # callbacks=[lr_cb],
            batch_print_steps=100,
            clip=10)

  with open('checkpoints/history.txt', 'w') as f:
    f.write(str(model.history))

if __name__ == '__main__':
  main()