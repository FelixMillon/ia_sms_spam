import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from collect_data import get_real_dataset

nltk.download("stopwords")

# Charger un sous-ensemble équilibré
texts, labels = get_real_dataset(n_ham=747, n_spam=747)
df = pd.DataFrame({"label": labels, "text": texts})

# Nettoyage et preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Supprimer les chiffres
    text = re.sub(r"[^a-z\s]", "", text)  # Supprimer la ponctuation
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])

# Réduction de dimension avec SVD
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

# Clustering KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=50, max_iter=500)
kmeans.fit(X_reduced)
y_kmeans = kmeans.labels_

# Mappage clusters -> labels REELLEMENT PRÉSENTS (post-analyse, pas d'utilisation dans l'algo)
cluster_map = defaultdict(lambda: "ham")
cluster_centers_labels = [
    (i, df[kmeans.labels_ == i]["label"].value_counts().idxmax())
    for i in range(2)
]
for i, label in cluster_centers_labels:
    cluster_map[i] = label

# Évaluation non supervisée (avec labels pour analyse post-clustering)
y_pred = np.array([cluster_map[label] for label in y_kmeans])
y_true = df["label"].values

# Évaluation
cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
silhouette = silhouette_score(X_reduced, y_kmeans)
report = classification_report(y_true, y_pred, target_names=["ham", "spam"])

# Affichage
print("Confusion Matrix:\n", cm)
print("\nSilhouette Score:", silhouette)
print("\nClassification Report:\n", report)

# Afficher quelques exemples par cluster
for i in range(2):
    print(f"\n=== Cluster {i} ===")
    examples = df[kmeans.labels_ == i].text.sample(10, random_state=42)
    print("\n".join(examples.values))
