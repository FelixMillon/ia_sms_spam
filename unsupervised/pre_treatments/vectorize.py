import math
import numpy as np

def l2_normalize(vec):
    norm = math.sqrt(sum(x**2 for x in vec))
    return [x / norm if norm > 0 else 0 for x in vec]

def normalize_tfidf(tfidf_matrix):
    return np.array([l2_normalize(vec) for vec in tfidf_matrix])