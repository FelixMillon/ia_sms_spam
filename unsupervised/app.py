from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# from collect.get_data import get_docs
# from pre_treatments.formatize import format_docs, reduce
from pre_treatments.clean import clean_data
from pre_treatments.terms_frequency import compute_tfidf
from pre_treatments.vectorize import normalize_tfidf
from pre_treatments.formatize import format_docs

# from separate.split import split_data
import pandas as pd
import json
import random
import numpy as np
import math
import re
import string
import json

### collect des données
def get_docs():
    return pd.read_csv("./spam.csv", encoding='latin-1')[['v1', 'v2']]

# recup données brutes
data = get_docs()

format_docs(data)

# enlever les données inutiles
data = clean_data(data)

# 3. Transformer les SMS en vecteurs numériques
tfidf_matrix = compute_tfidf(data['clean_message'])
tfidf_normalized = normalize_tfidf(tfidf_matrix)
