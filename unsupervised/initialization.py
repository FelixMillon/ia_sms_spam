from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN

def vectorize_messages(clean_messages):
    """
    Transforme les messages nettoyés en vecteurs TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_messages)
    return X, vectorizer

def init_kmeans(X, n_clusters=2, random_state=42):
    """
    Initialise et retourne un modèle KMeans.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)
    return model

def init_dbscan(X, eps=0.5, min_samples=5):
    """
    Initialise et retourne un modèle DBSCAN.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model
