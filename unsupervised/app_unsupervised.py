import pandas as pd
import numpy as np
import re
import nltk
import hdbscan
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collect_data import get_real_dataset

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Charger un sous-ensemble équilibré de données labellisées
texts, labels = get_real_dataset(n_ham=747, n_spam=747)
df = pd.DataFrame({"text": texts, "label": labels})

# Nettoyage et preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Supprimer les chiffres
    text = re.sub(r"[^a-z\s]", "", text)  # Supprimer la ponctuation
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])

# Réduction de dimension
svd = TruncatedSVD(n_components=150, random_state=42)
X_reduced = svd.fit_transform(X)

# Clustering avec HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='manhattan')
y_hdb = clusterer.fit_predict(X_reduced)

# Marquer les clusters
df["cluster"] = y_hdb
mask = y_hdb != -1

if np.sum(mask) == 0:
    print("\n❌ Aucun cluster détecté par HDBSCAN (tous les points sont du bruit).")
else:
    silhouette = silhouette_score(X_reduced[mask], y_hdb[mask])
    print("\nSilhouette Score:", silhouette)

    # Analyse par cluster
    unique_clusters = sorted(df["cluster"].unique())
    for i in unique_clusters:
        subset = df[df["cluster"] == i]
        print(f"\n=== Cluster {i} ===")
        samples = subset["text"].sample(min(10, len(subset)), random_state=42)
        print("\n".join(samples.values))

        top_words = pd.Series(" ".join(subset["clean_text"]).split()).value_counts().head(10)
        print(f"\nTop words in Cluster {i}:")
        for word, freq in top_words.items():
            print(f"{word}: {freq}")

    # Évaluation non supervisée (avec les labels, pour analyse seulement)
    from sklearn.metrics import confusion_matrix, classification_report
    from collections import defaultdict

    cluster_map = defaultdict(lambda: "ham")
    for i in unique_clusters:
        true_labels = df[df["cluster"] == i]["label"]
        if not true_labels.empty:
            cluster_map[i] = true_labels.value_counts().idxmax()

    y_pred = df["cluster"].map(cluster_map)
    y_true = df["label"]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["ham", "spam"]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))
