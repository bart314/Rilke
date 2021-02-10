import gensim
import numpy as np

seeds = ['Erstaunte', 'Aber', 'Nirgends', 'O', 'Hiersein', 'Nicht', 'Hier', 'Stehn', 'Jeder', 'Fänden', 'Eines', 'Plätze', 'Werbung', 'Einsam', 'Oder', 'Da', 'Preise', 'Ich', 'Ach', 'Mit', 'Dass', 'Frühe', 'Und', 'Engel', 'Siehe', 'Dann', 'Freilich', 'Wo', 'Feigenbaum', 'Liebende', 'Schließlich', 'Doch', 'Wunderlich', 'Nur', 'Uns', 'Denn', 'War', 'Wießt', 'Ja', 'Hab', 'Oh', 'Wir', 'Jede', 'Wäre', 'Warum', 'Würfen', 'Stimmen', 'Erde', 'Du', 'Wer']

def get_seed_word():
  np.random.shuffle(seeds)
  return seeds.pop()

def get_line_length():
  lens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25]
  dist = [0.01839080459770115, 0.006896551724137931, 0.009195402298850575, 0.03218390804597701, 0.07241379310344828, 0.12528735632183907, 0.2, 0.20229885057471264, 0.167816091954023, 0.08850574712643679, 0.05172413793103448, 0.013793103448275862, 0.0034482758620689655, 0.0034482758620689655, 0.0022988505747126436, 0.0022988505747126436]
  return np.random.choice(lens, p=dist)

def get_strophe_length():
  lens = [8, 7, 10, 4, 3, 5, 6, 9, 11, 13, 12, 2, 15, 18, 19, 28, 31, 14, 17, 21, 22, 24, 25, 32] 
  dist = [0.19047619047619047, 0.09523809523809523, 0.09523809523809523, 0.07142857142857142, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.03571428571428571, 0.023809523809523808, 0.023809523809523808, 0.023809523809523808, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904, 0.011904761904761904]
  return np.random.choice(lens, p=dist)




def find_ten_closest_words(v, k=1):
        # Calculate the vector difference from each word to the input vector
    diff = embedding.values - v 
    # Get the norm of each difference vector. 
    # It means the squared euclidean distance from each word to the input vector
    delta = np.sum(diff * diff, axis=1)
    # Find the index of the minimun distance in the array
    # i = np.argmin(delta)
    # Return the row name for this item
    print (delta[:10])
    # from https://numpy.org/doc/stable/reference/generated/numpy.argsort.html#numpy.argsort
    # 
    ind = delta.argsort()
    print (ind[:10])
    print( [embedding.iloc[i].name for i in ind[:10]])


def get_synoniems(w):
    syns = trained_model.most_similar(w)
    print (f"Bitte wählen Sie eine Synoniem für {w}")
    ctr = 1
    for s in syns:
        print (f'[{ctr}] {s[0]}')
        ctr += 1

    ch = input ('Ihre Wähl: ')
    return syns[int(ch)-1][0]



trained_model = gensim.models.KeyedVectors.load_word2vec_format('files/german.model', binary=True)
## remove original vectors to free up memory
trained_model.init_sims(replace=True)
