import re
import string

# clean des données
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_data(data):
    data['clean_message'] = data['message'].apply(clean_text)
    return data[data['clean_message'].str.strip() != '']