"""
app.py
------
Task 6: Streamlit UI
Task 7: LLM integration wiring

Run with: streamlit run app.py
"""

import os
import sys
import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import preprocess, extract_suspicious_signals
from llm_explainer import explain_prediction

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Spam Email Detector", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "spam_model.joblib"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    return model, vectorizer


st.title("🛡️ Spam Email Detector")
st.caption("ML-based spam classification, explained by an LLM.")

with st.sidebar:
    st.header("About")
    st.write(
        "The Machine Learning model (TF-IDF + Naive Bayes / Logistic Regression) "
        "makes the Spam/Ham decision. The LLM only explains that decision -- "
        "it never re-classifies the message."
    )
    st.write(f"**LLM Provider:** `{os.environ.get('LLM_PROVIDER', 'groq')}`")
    st.write("Set `GROQ_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY` as an environment "
             "variable to enable live AI explanations. Without a key, a rule-based "
             "fallback explanation is shown so the app still works for demos.")

message = st.text_area("Paste an email or SMS message:", height=180,
                        placeholder="e.g. Congratulations! You've won a $1000 gift card. Click here to claim now!")

analyze_clicked = st.button("Analyze", type="primary")

if analyze_clicked:
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        try:
            model, vectorizer = load_artifacts()
        except FileNotFoundError:
            st.error(
                "No trained model found. Please run `python src/train_model.py` first "
                "to generate models/spam_model.joblib and models/tfidf_vectorizer.joblib."
            )
            st.stop()

        cleaned = preprocess(message)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        confidence = proba[classes.index(prediction)] * 100

        signals = extract_suspicious_signals(message)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if prediction == "spam":
                st.error(f"### 🚫 SPAM")
            else:
                st.success(f"### ✅ NOT SPAM (Ham)")
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        with st.spinner("Generating AI explanation..."):
            result = explain_prediction(message, prediction, confidence, signals)

        st.subheader("🤖 AI Explanation")
        st.write(result.get("explanation", ""))

        st.subheader("🔍 Suspicious Indicators")
        phrases = result.get("suspicious_phrases", [])
        if phrases:
            for p in phrases:
                st.markdown(f"- {p}")
        else:
            st.write("None flagged.")

        st.subheader("🛡️ Safety Recommendations")
        for tip in result.get("safety_tips", []):
            st.markdown(f"- {tip}")

        st.subheader("➡️ Suggested Next Action")
        st.info(result.get("recommended_action", ""))
