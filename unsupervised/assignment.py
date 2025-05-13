# assignment.py

def assign_clusters(model, messages):
    """
    Associe chaque message à un cluster à l'aide du modèle KMeans.
    Retourne une liste de tuples (message, cluster).
    """
    labels = model.labels_
    return list(zip(messages, labels))
