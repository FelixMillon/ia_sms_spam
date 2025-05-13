import numpy as np

from initialisazion.centroids import initialize_centroids
from assignation.clusters import assign_clusters
from updates.centroids import update_centroids

def kmeans(X, k=2, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids