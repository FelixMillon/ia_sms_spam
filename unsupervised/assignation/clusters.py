import numpy as np

def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [np.linalg.norm(x - centroid) for centroid in centroids]
        cluster_id = np.argmin(distances)
        clusters.append(cluster_id)
    return clusters