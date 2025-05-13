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
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords')

# Charger les données
try:
    df = pd.read_csv("data/spam.csv", encoding="latin-1")
except FileNotFoundError:
    df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "text"]
df = df[df["label"].isin(["ham", "spam"])]

# Équilibrage des classes
n = df["label"].value_counts().min()
ham = df[df["label"] == "ham"].sample(n=n, random_state=42)
spam = df[df["label"] == "spam"].sample(n=n, random_state=42)
df = pd.concat([ham, spam]).sample(frac=1, random_state=42).reset_index(drop=True)


# Nettoyage et preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    text = re.sub(r'[^a-z\s]', '', text)  # Supprimer la ponctuation
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
X = vectorizer.fit_transform(df["clean_text"])

# Réduction de dimension avec SVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)


X_scaled = StandardScaler().fit_transform(X_reduced)


# Clustering KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)
y_kmeans = kmeans.labels_

# Mappage clusters -> labels
cluster_map = defaultdict(lambda: "ham")
cluster_centers_labels = [
    (i, df[kmeans.labels_ == i]["label"].value_counts().idxmax())
    for i in range(2)
]
for i, label in cluster_centers_labels:
    cluster_map[i] = label

# Conversion des clusters en prédictions de classes
y_pred = np.array([cluster_map[label] for label in y_kmeans])
y_true = df["label"].values

# Évaluation
cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
silhouette = silhouette_score(X_scaled, y_kmeans)
report = classification_report(y_true, y_pred, target_names=["ham", "spam"])

# Affichage
print("Confusion Matrix:\n", cm)
print("\nSilhouette Score:", silhouette)
print("\nClassification Report:\n", report)

# Afficher quelques exemples
for i in range(2):
    print(f"\nCluster {i} composition:")
    print(df[kmeans.labels_ == i]["label"].value_counts())
