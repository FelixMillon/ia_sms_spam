
import pandas as pd
import json
from collections import Counter

from pre_treatments.clean import clean_data
from pre_treatments.terms_frequency import compute_tfidf
from pre_treatments.vectorize import normalize_tfidf
from pre_treatments.formatize import format_docs
from modele.kmeans import kmeans


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# from separate.split import split_data

def get_docs():
    return pd.read_csv("./spam.csv", encoding='latin-1')[['v1', 'v2']]

print("gathering data")
data = get_docs()

print("formating data")
format_docs(data)

print("cleaning data")
data = clean_data(data)

print("analyse frequency of terms")
tfidf_matrix = compute_tfidf(data['clean_message'])

print("normalize frenquencies (as vectors)")
tfidf_normalized = normalize_tfidf(tfidf_matrix)


print("training")
k = 2
clusters, centroids = kmeans(tfidf_normalized, k=k)


print("Attribution des étiquettes aux clusters...")
cluster_labels = {}

for cluster_id in set(clusters):
    cluster_messages = data[clusters == cluster_id]
    label_counts = Counter(cluster_messages['label'])
    most_common_label = label_counts.most_common(1)[0][0]
    cluster_labels[cluster_id] = most_common_label


print("Évaluation de l'exactitude du clustering...")
correct_predictions = 0
total_predictions = len(data)

for i, cluster_id in enumerate(clusters):
    predicted_label = cluster_labels[cluster_id]
    actual_label = data['label'].iloc[i]
    if predicted_label == actual_label:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions

print(f"Accuracy du modèle K-Means: {accuracy * 100:.2f}%")
# # Construction du résumé des clusters
# cluster_summary = {}
# for i, cluster_id in enumerate(clusters):
#     cluster_id = int(cluster_id)  # Assure-toi que la clé est un int standard
#     if cluster_id not in cluster_summary:
#         cluster_summary[cluster_id] = []
#     cluster_summary[cluster_id].append(data['clean_message'].iloc[i])

# # Export en JSON (avec clés au bon format)
# with open('cluster_summary.json', 'w', encoding='utf-8') as f:
#     json.dump(cluster_summary, f, ensure_ascii=False, indent=2)

# cluster_summary = {}
# for i, cluster_id in enumerate(clusters):
#     if cluster_id not in cluster_summary:
#         cluster_summary[cluster_id] = []
#     cluster_summary[cluster_id].append(data['clean_message'].iloc[i])

# with open('cluster_summary.json', 'w') as f:
#     json.dump(cluster_summary, f)

# for cluster_id, messages in cluster_summary.items():
#     print(f"Cluster {cluster_id} ({cluster_labels[cluster_id]}):")
#     print(messages[:5])