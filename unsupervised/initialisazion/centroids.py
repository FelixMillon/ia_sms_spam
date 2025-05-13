import random

def initialize_centroids(X, k):
    indices = random.sample(range(len(X)), k)
    centroids = X[indices]
    return centroids