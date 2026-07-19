"""
preprocessing.py
----------------
Task 2: Text Preprocessing

Cleans raw email/SMS text so it can be turned into numeric features:
- lowercase
- remove punctuation / special characters / numbers
- tokenize
- remove stopwords
- lemmatize
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (only runs once, then cached)
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits/urls, return cleaned raw string."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # urls
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"[^a-z\s]", " ", text)                   # punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()                # extra whitespace
    return text


def preprocess(text: str) -> str:
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
    Returns a single space-joined string, ready for TF-IDF.
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def extract_suspicious_signals(text: str) -> dict:
    """
    Lightweight, rule-based signal extraction used to enrich the LLM prompt.
    This is NOT used for classification -- only to give the LLM concrete,
    grounded evidence to talk about (so it isn't just guessing).
    """
    raw = str(text)
    signals = {
        "has_url": bool(re.search(r"http\S+|www\.\S+", raw, re.I)),
        "has_phone_number": bool(re.search(r"\b\d{5,}\b", raw)),
        "has_money_symbol": bool(re.search(r"[$£€]|(?:\bfree\b)|(?:\bwin\b)|(?:\bprize\b)", raw, re.I)),
        "excessive_caps": sum(1 for w in raw.split() if w.isupper() and len(w) > 2) >= 3,
        "excessive_punctuation": bool(re.search(r"[!?]{2,}", raw)),
        "urgency_words": bool(re.search(
            r"\b(urgent|immediately|now|act now|limited time|expire|verify|suspend|congratulations|winner)\b",
            raw, re.I)),
    }
    return signals


if __name__ == "__main__":
    sample = "WINNER!! You have been selected to receive a £900 prize. Call now! http://bit.ly/claim"
    print("Cleaned:", preprocess(sample))
    print("Signals:", extract_suspicious_signals(sample))
