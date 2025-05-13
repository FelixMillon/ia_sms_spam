
import pandas as pd

def get_real_dataset(n_ham=250, n_spam=250):
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df = df[df["label"].isin(["ham", "spam"])]
    
    ham = df[df["label"] == "ham"].sample(n=747, random_state=42)
    spam = df[df["label"] == "spam"]
    df = pd.concat([ham, spam]).sample(frac=1, random_state=42).reset_index(drop=True)

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df['text'].tolist(), df['label'].tolist()

