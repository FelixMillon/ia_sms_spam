import random
import numpy as np

def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[np.array(clusters) == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = X[random.randint(0, len(X)-1)]
        new_centroids.append(new_centroid)
    return np.array(new_centroids)