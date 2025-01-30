import pickle
import numpy as np

from dotenv import load_dotenv; load_dotenv()
import sys, os; sys.path.append(os.getenv('MP'))

import femto_flow as ff

def load(path):
  with open(path, 'rb') as f:
    return pickle.load(f)
  
def prepare_input(input, vectorizer, seq_len):
  output = np.array(vectorizer.transform(input.lower()))
  output = np.pad(output, ((seq_len-output.shape[0], 0), (0,0))).reshape(1, -1)
  
  # (n_batches, batch_size, seq_len)
  return output[np.newaxis,:,:]

def main():
  seq_len = 8
  gen_length = 100
  
  # Import network and vectorizer
  model = load(r"demos/c_generative_demo/checkpoints/8.pkl")
  vectorizer = load(r"demos/c_generative_demo/checkpoints/word_processor.pkl")
  
  raw_input = "at are y"
  input = prepare_input(raw_input, vectorizer, seq_len)
  
  output = []
  for i in range(gen_length):
    pred = np.argmax(model.predict(input))
    
    # vectorizer excepts a corpus of documents, not a single token, hence the [[]]
    pred_token = vectorizer.inverse_transform([[pred]])
    
    output.append(pred_token)
    
    # shift input to the left and add pred to end
    input[:,:,:-1] = input[:,:,1:]
    input[:,:,-1] = pred

  # print(''.join([t for t in output]))
  print(''.join([t[0][0] for t in output]))
  
if __name__ == '__main__':
  main()