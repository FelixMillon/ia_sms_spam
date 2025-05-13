import pandas as pd

def get_real_dataset(n_ham=250, n_spam=250):
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df = df[df["label"].isin(["ham", "spam"])]

    ham = df[df["label"] == "ham"].sample(n=n_ham, random_state=42)
    spam = df[df["label"] == "spam"].sample(n=n_spam, random_state=42)
    df = pd.concat([ham, spam]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Ne retourne pas les labels numériques (non supervisé)
    return df["text"].tolist(), df["label"].tolist()