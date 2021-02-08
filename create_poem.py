import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle as pkl
import logging

from utils import *

logging.basicConfig(filename='network.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

model = load_model('files/rilke_model.h5')
# PREDICTION OF THE NEXT WORD
with open('files/tokenizer.pkl', 'rb') as t:
  tokenizer = pkl.load(t)

total_words = len(tokenizer.word_index) + 1 # for OOV token
logging.info (f'The total of different words in the corpus is {total_words}')


#tokenizer = Tokenizer()
#data = open('data/el1.txt','r').read()
#corpus = data.lower().split('\n') 
#tokenizer.fit_on_texts(corpus) 

# STAPPENPLAN
# Bepaal aantal stroves
# Per strove:
#   Bepaal seed-word
#   Bepaal aantal regels
#   Per regel:
#      Bepaal aantal woorden
#       

lengtes = [10, 7, 5, 8, 13, 6, 9, 7, 7, 12]
max_sequence_len = 13 
elegie_length = np.random.choice(lengtes) 
logging.info (f"Creating an elegie with {elegie_length} strophes.")

for s in range(elegie_length): 
  s_length = 4 #get_strophe_length()
  line = get_seed_word() 
  logging.info (f'Strophe with {s_length} lines, using {line} as start word...')
  for _ in range(s_length):
    l_length = get_line_length()
    logging.info (f'line with a length of {l_length} words...')

    for _ in range(l_length):
      token_list = tokenizer.texts_to_sequences([line])[0]
      token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
      #predicted = model.predict_classes(token_list, verbose=0)
      predicted = np.argmax(model.predict(token_list), axis=-1)
      output_word = ''

      #reverse look-up
      for word, index in tokenizer.word_index.items():
        if index == predicted:
          output_word = word
          break
      line += ' ' + output_word

    line += "\n"
  
  print (line) 
  print ("\n")
# kan dit niet makkelijker?


# BIGGER CORPUS
# dimensionality of the embedding (100)
# number of LSTM to 150
# Optimizer Adam(lr=0.01)
# epochs 





