import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense

import numpy as np
import pickle as pkl

# STEP 1: 
# tokenize the corpus

tokenizer = Tokenizer()
data = ''
for i in range(1,11):
    data += open(f'data/el{i}.txt','r').read()
corpus = data.split('\n')

tokenizer.fit_on_texts(corpus) 
with open ('files/tokenizer.pkl', 'wb') as f:
  pkl.dump(tokenizer, f, protocol=pkl.HIGHEST_PROTOCOL)

# STEP 2: 
# create a dictionary of words 
# key-value pair, key => word, value => token for that word

total_words = len(tokenizer.word_index) + 1 # for OOV token
print (f'The total of different words in the corpus is {total_words}')

# STEP 3:
# Generating the training data

print ('=================')
print ('Generating input-sequences')
input_sequences = []
for line in corpus:
  #list of the token representing the words
  token_list = tokenizer.texts_to_sequences([line])[0]

  #first two words, first three words, first four words, ...
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

print (f'The length of the input_sequence is {len(input_sequences)}')


# STEP 5:
# Find the largest sentence in the corpus and 
# pad all the other sentences to this length

# find the length of the longest sentence in the corpus
print ('Finding the length of the longest sentence in the corpus')
max_sequence_length = max([len(x) for x in input_sequences])
print (f'The longest sentence in the corpus is {max_sequence_length } words.')

#pad all the sequences so that they are the same length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')) 
#padded input sequences

#turn them into vectors of x's and y's, input values and their labels...
#last character is the label, first n characters are the x's

# STEP 6:
# Creating train and validation data-sets and
# making X-matrices and y-vector
m,n = input_sequences.shape 
data_train, data_cv = input_sequences[:int(m*.8)], input_sequences[int(m*.8):]
X_train, X_cv = data_train[:,:-1], data_cv[:,:-1]
labels_train, labels_cv = data_train[:,-1], data_cv[:,-1]
# STEP 7:
# one-hot encode the labels
ys_train = tf.keras.utils.to_categorical(labels_train, num_classes=total_words)
ys_cv = tf.keras.utils.to_categorical(labels_cv, num_classes=total_words)


# STEP 8:
# Create a model in order to
# find out what the next word should be
model = Sequential() 
model.add(Embedding(total_words, 64, input_length=max_sequence_length - 1))
model.add(LSTM(200))

# One neuron per word, which will light up if that is the predicted word
model.add(Dense(total_words, activation="softmax"))

# STEP 9:
# Compile and train the model 

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
hist = model.fit(X_train, ys_train, epochs=100, validation_data=[X_cv, ys_cv])
print ("=== [SAVE HISTORY] =====")
with open('history.pkl', 'wb') as file_pi:
    pkl.dump(hist.history, file_pi)

print ('==== [SAVE MODEL] ====')
model.save('files/rilke_model.h5')



# STEP 10 (optional):
# prediction: a lot of repetitions, because the LSTM only carries information forward
# Make the LSTM bidirectional

#model.add(Bidirectional(LSTM(20))
