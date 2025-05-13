import numpy as np
import math

# Nombre d'occurrences du mot / nombre total de mots
def compute_tf(message, vocabulary):
    tf = {}
    words = message.split()
    total_words = len(words)
    # if total_words != 0:
    for word in vocabulary:
        tf[word] = words.count(word) / total_words
    return tf

# Calcul de l'inverse de fréquence du mot
def compute_idf(documents, vocabulary):
    N = len(documents)
    idf = {}
    for word in vocabulary:
        frenquency = 0
        for doc in documents:
            if word in doc.split():
                frenquency += 1
        idf_value = math.log(N / (frenquency + 1))
        idf[word] = idf_value
    return idf

# Calcul du rapport occurence fréquence
def compute_tfidf(documents):
    vocabulary = set()
    for message in documents:
        vocabulary.update(message.split())

    idf = compute_idf(documents, vocabulary)

    tfidf_matrix = []
    
    for message in documents:
        tf = compute_tf(message, vocabulary)
        tfidf = []
        for word in vocabulary:
            tfidf.append(tf.get(word, 0) * idf[word])
        tfidf_matrix.append(tfidf)

    return np.array(tfidf_matrix)