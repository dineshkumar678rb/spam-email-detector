"""
train_model.py
---------------
Task 3: Feature Extraction (TF-IDF)
Task 4: Machine Learning Model (Naive Bayes vs Logistic Regression)
Task 5: Model Evaluation

Trains both models, compares them, and saves the best one + its vectorizer
to models/ for use by the Streamlit app.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from eda import load_data, run_eda
from preprocessing import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshots")


def evaluate(name, y_true, y_pred):
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="spam"),
        "recall": recall_score(y_true, y_pred, pos_label="spam"),
        "f1": f1_score(y_true, y_pred, pos_label="spam"),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k.capitalize():10s}: {v:.4f}")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(SCREENSHOTS_DIR, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"))
    plt.close()

    return metrics


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Task 1 (re-used)
    df = load_data()
    df = run_eda(df)

    # Task 2
    print("\nPreprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess)

    # Task 3
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # Task 4: train both models
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Task 5: evaluate both
    nb_metrics = evaluate("Naive Bayes", y_test, nb_preds)
    lr_metrics = evaluate("Logistic Regression", y_test, lr_preds)

    # Pick the best model by F1-score (best metric for imbalanced spam/ham data)
    best_name, best_model, best_metrics = (
        ("Naive Bayes", nb_model, nb_metrics)
        if nb_metrics["f1"] >= lr_metrics["f1"]
        else ("Logistic Regression", lr_model, lr_metrics)
    )
    print(f"\n>>> Best model selected: {best_name} (F1={best_metrics['f1']:.4f})")

    # Save best model + vectorizer + a comparison report
    joblib.dump(best_model, os.path.join(MODELS_DIR, "spam_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    comparison = pd.DataFrame([nb_metrics, lr_metrics])
    comparison.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    print(f"\nSaved model, vectorizer, and comparison report to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
