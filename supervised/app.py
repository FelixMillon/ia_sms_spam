from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# from frequency.terms_frequency import compute_tfidf
# from collect.get_data import get_docs
# from pre_treatments.formatize import format_docs, reduce
# from pre_treatments.clean import clean_data
# from separate.split import split_data
import pandas as pd
import json
import random
import numpy as np
import math
import re
import string
import nltk
import json
from nltk.corpus import stopwords

nltk.download('stopwords')

### generate stop word
stop_words_english = stopwords.words('english')
stop_words_french = stopwords.words('french')

stopwords_dict = {
    'english': stop_words_english,
    'french': stop_words_french
}

with open('./stop_words.json', 'w', encoding='utf-8') as f:
    json.dump(stopwords_dict, f, ensure_ascii=False, indent=4)


### collect des données
def get_docs():
    return pd.read_csv("./spam.csv", encoding='latin-1')[['v1', 'v2']]


def get_stop_words():
    with open('stop_words.json', 'r', encoding='utf-8') as f:
        stopwords_dict = json.load(f)
    return stopwords_dict

### clean

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in STOP_WORDS])
    return text

def clean_data(data):
    data['clean_message'] = data['message'].apply(clean_text)
    return data[data['clean_message'].str.strip() != '']

### formatize

def format_docs(data):
    data.columns = ['label', 'message']

def reduce(data):
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

### frequency

# Nombre d'occurrences du mot / nombre total de mots
def compute_tf(message, vocabulary):
    tf = {}
    words = message.split()
    total_words = len(words)
    # if total_words != 0:
    for word in vocabulary:
        tf[word] = words.count(word) / total_words
    return tf

# Calcul de l'inverse de fréquence du mot
def compute_idf(documents, vocabulary):
    N = len(documents)
    idf = {}
    for word in vocabulary:
        frenquency = 0
        for doc in documents:
            if word in doc.split():
                frenquency += 1
        idf_value = math.log(N / (frenquency + 1))
        idf[word] = idf_value
    return idf

# Calcul du rapport occurence fréquence
def compute_tfidf(documents):
    vocabulary = set()
    for message in documents:
        vocabulary.update(message.split())

    idf = compute_idf(documents, vocabulary)

    tfidf_matrix = []
    
    for message in documents:
        tf = compute_tf(message, vocabulary)
        tfidf = []
        for word in vocabulary:
            tfidf.append(tf.get(word, 0) * idf[word])
        tfidf_matrix.append(tfidf)

    return np.array(tfidf_matrix)

### separate
def shuffle_data(messages, labels):
    data = list(zip(messages, labels))
    random.shuffle(data)
    messages_shuffled, labels_shuffled = zip(*data)
    return messages_shuffled, labels_shuffled

def split_data(messages, labels, test_size=0.2):
    messages_shuffled, labels_shuffled = shuffle_data(messages, labels)

    test_size_int = int(len(messages) * test_size)

    X_train = messages_shuffled[:-test_size_int]  # 80% pour l'entraînement
    y_train = labels_shuffled[:-test_size_int]

    X_test = messages_shuffled[-test_size_int:]  # 20% pour le test
    y_test = labels_shuffled[-test_size_int:]

    return X_train, X_test, y_train, y_test



stopwords_dict = get_stop_words()
STOP_WORDS = stopwords_dict['english']

# recup données brutes
data = get_docs()

# formater et reduire
format_docs(data)
reduce(data)

# enlever les données inutiles
data = clean_data(data)

# 3. Transformer les SMS en vecteurs numériques
X = compute_tfidf(data['clean_message'])
y = data['label']

# préparer les données de test et de train
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

# entrainement
model = MultinomialNB()
model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# enregistrement des spams
spam_indices = [i for i, pred in enumerate(y_pred) if pred == 1]

spam_original_messages = [data['message'].iloc[i] for i in spam_indices]
spam_true_labels = [y_test[i] for i in spam_indices]
spam_pred_labels = [y_pred[i] for i in spam_indices]

with open("spam_messages_with_predictions.txt", "w", encoding="utf-8") as file:
    for msg, true_label, pred_label in zip(spam_original_messages, spam_true_labels, spam_pred_labels):
        if true_label == pred_label:
            result = "Correct"
        else:
            result = "Incorrect"
        
        file.write(f"Message: {msg}\nPredicted: {'spam' if pred_label == 1 else 'ham'}\nTrue Label: {'spam' if true_label == 1 else 'ham'}\nPrediction: {result}\n\n")
