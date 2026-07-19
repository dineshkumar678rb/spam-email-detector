"""
eda.py
------
Task 1: Dataset Exploration

Loads the SMS Spam Collection dataset and produces:
- class distribution
- missing/duplicate checks
- word clouds for spam vs ham
- most common words bar chart

Expected input file: data/spam.csv with columns 'v1' (label) and 'v2' (text),
which is the standard format for this dataset on both UCI and Kaggle.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

from preprocessing import preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "spam.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshots")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    # The raw Kaggle/UCI file has extra unnamed columns + short column names
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna()
    return df


def run_eda(df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Shape:", df.shape)
    print("\nClass distribution:\n", df["label"].value_counts())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # Drop duplicates for downstream training
    df = df.drop_duplicates()

    # --- Class distribution chart ---
    plt.figure(figsize=(5, 4))
    df["label"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Spam vs Ham distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"))
    plt.close()

    # --- Word clouds ---
    for label in ["spam", "ham"]:
        text_blob = " ".join(df.loc[df["label"] == label, "text"].apply(preprocess))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Most common words: {label.upper()}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"wordcloud_{label}.png"))
        plt.close()

    # --- Top word frequency bar chart (spam only) ---
    spam_tokens = " ".join(df.loc[df["label"] == "spam", "text"].apply(preprocess)).split()
    common = Counter(spam_tokens).most_common(15)
    words, counts = zip(*common)
    plt.figure(figsize=(8, 5))
    plt.barh(words[::-1], counts[::-1], color="#C44E52")
    plt.title("Top 15 words in spam messages")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top_spam_words.png"))
    plt.close()

    print(f"\nCharts saved to {OUT_DIR}/")
    return df


if __name__ == "__main__":
    data = load_data()
    run_eda(data)
