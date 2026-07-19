# 🛡️ Spam Email Detector with AI Email Explanation

Classifies an email/SMS as **Spam** or **Ham** using a TF-IDF + Machine Learning
pipeline, then uses an LLM to explain *why* — without letting the LLM do the
actual classification.

## 🚀 Live Demo

**Try it here:** https://spam-email-detector-y4n8xrbaupchshegy7w7fe.streamlit.app

> Note: this runs on a free-tier server, so if it hasn't been visited in a
> while it may take 10-20 seconds to "wake up" on first load.

## 📊 Results

Trained and evaluated on the SMS Spam Collection dataset (5,572 messages):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Naive Bayes (selected)** | 96.8% | 99.0% | 75.6% | **85.7%** |
| Logistic Regression | 95.6% | 98.9% | 65.7% | 78.9% |

Naive Bayes was selected as the best model based on F1-score, which better
reflects performance on this dataset's class imbalance (747 spam vs. 4,825
ham messages) than raw accuracy alone.

## Project Structure

```
spam_detector/
├── app.py                     # Streamlit web app (Task 6 + 7)
├── requirements.txt
├── data/
│   └── spam.csv                # SMS Spam Collection dataset
├── models/                     # created after training
│   ├── spam_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_comparison.csv
├── screenshots/                # EDA charts + confusion matrices
└── src/
    ├── preprocessing.py        # Task 2: text cleaning
    ├── eda.py                  # Task 1: dataset exploration
    ├── train_model.py          # Tasks 3-5: TF-IDF, model training, evaluation
    └── llm_explainer.py        # Task 7: LLM integration
```

## Setup (running it locally)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Get the dataset**

Already included at `data/spam.csv`. If you want to source it yourself:
- UCI: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

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

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · NLTK · Streamlit · Groq API (LLM)

## Notes

- The LLM is deliberately kept **out** of the classification decision — it only
  explains and advises on what the ML model already decided, per the project
  design.
- `models/model_comparison.csv` has the full side-by-side metrics for both
  algorithms.
- Charts in `screenshots/` cover dataset exploration (word clouds, class
  distribution) and model evaluation (confusion matrices).
