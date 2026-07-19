# Spam Email Detector with AI Email Explanation

Classifies an email/SMS as **Spam** or **Ham** using a TF-IDF + Machine Learning
pipeline, then uses an LLM to explain *why* — without letting the LLM do the
actual classification.

## Project Structure

```
spam_detector/
├── app.py                     # Streamlit web app (Task 6 + 7)
├── requirements.txt
├── data/
│   └── spam.csv                # <-- you add this (see Setup step 2)
├── models/                     # created after training
│   ├── spam_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_comparison.csv
├── screenshots/                # EDA charts + confusion matrices land here
└── src/
    ├── preprocessing.py        # Task 2: text cleaning
    ├── eda.py                  # Task 1: dataset exploration
    ├── train_model.py          # Tasks 3-5: TF-IDF, model training, evaluation
    └── llm_explainer.py        # Task 7: LLM integration
```

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download the dataset**

Get the SMS Spam Collection dataset from either:
- UCI: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Save it as `data/spam.csv`. It should have columns `v1` (label: spam/ham) and
`v2` (the message text) — this is the standard layout for both sources.

**3. Set your LLM API key**

Pick one provider and export the matching key as an environment variable:

```bash
# Groq (fast, generous free tier)
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_key_here

# OR OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here

# OR Gemini
export LLM_PROVIDER=gemini
export GEMINI_API_KEY=your_key_here
```

> If you skip this step, the app still runs — it falls back to a rule-based
> explanation so you can demo the ML side without an API key.

**4. Train the model**
```bash
cd src
python train_model.py
```
This runs the EDA (Task 1), preprocessing (Task 2), TF-IDF vectorization
(Task 3), trains both Naive Bayes and Logistic Regression, evaluates them
(Task 5), and saves whichever model scores higher on F1 to `models/`.

Charts (class distribution, word clouds, confusion matrices) are saved to
`screenshots/` — use these directly in your project report.

**5. Run the app**
```bash
cd ..
streamlit run app.py
```

Paste a message, click **Analyze**, and you'll see:
- Spam / Ham prediction + confidence score
- AI-generated explanation of the prediction
- Suspicious phrases/keywords
- Safety recommendations
- Suggested next action

## How it fits together

```
Message → preprocessing.py (clean) → TF-IDF vectorizer → ML model → Spam/Ham + confidence
                                                                          │
                                                                          ▼
                                                      llm_explainer.py (explains the
                                                      ML model's decision — does NOT
                                                      re-classify)
```

## Notes for your report

- `model_comparison.csv` (in `models/` after training) has side-by-side
  Accuracy/Precision/Recall/F1 for both algorithms — good for the "model
  performance" section of your deliverable.
- Screenshots in `screenshots/` cover both the EDA (Task 1) and evaluation
  (Task 5) requirements.
- The design deliberately keeps the LLM **out** of the classification decision,
  per the project spec — it only explains and advises.
