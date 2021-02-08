import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

model = load_model('rilke_model.h5')
# PREDICTION OF THE NEXT WORD
tokenizer = Tokenizer()
data = open('data/el1.txt','r').read()
corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus) 
seed_text = "Wer"
next_words = 25
max_sequence_len = 13

#reverse look-up
for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
  predicted = model.predict_classes(token_list, verbose=0)
  #predicted = np.argmax(model.predict(token_list), axis=-1)
  #print (model.predict(token_list))
  output_word = ''
  #print (tokenizer.word_index.items())
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word
      break
  seed_text += ' ' + output_word

print (seed_text)

# kan dit niet makkelijker?


# BIGGER CORPUS
# dimensionality of the embedding (100)
# number of LSTM to 150
# Optimizer Adam(lr=0.01)
# epochs 





