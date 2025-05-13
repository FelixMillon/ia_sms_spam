import numpy as np
from initialisazion.centroids import initialize_centroids
from assignation.clusters import assign_clusters
from updates.centroids import update_centroids

def kmeans(X, k=2, max_iters=100):
    """
    Fonction KMeans qui retourne les clusters et les centroids.
    :param X: données à clustériser
    :param k: nombre de clusters
    :param max_iters: nombre d'itérations maximal
    """
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        print(f"iteration {i}")
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids