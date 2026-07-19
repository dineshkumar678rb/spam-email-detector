"""
llm_explainer.py
----------------
Task 7: Integrate an LLM to explain the ML model's prediction.

IMPORTANT: The LLM never classifies the message. The ML model (Naive Bayes /
Logistic Regression) already decided Spam vs Ham. This module only asks the
LLM to explain that decision in plain language, point out suspicious phrases,
and give safety advice.

Supports three providers -- pick one by setting LLM_PROVIDER in your
environment (or .env file): "groq", "openai", or "gemini".
Only one API key is required, matching whichever provider you choose.
"""

import os
import json
import requests

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()  # groq | openai | gemini

PROMPT_TEMPLATE = """You are a cybersecurity assistant helping a non-technical user understand
why a message was flagged by a machine learning spam filter.

The ML model has ALREADY made this decision -- do not re-classify the message yourself,
just explain and advise.

Message:
\"\"\"{message}\"\"\"

ML Model Prediction: {prediction}
Model Confidence: {confidence:.1f}%

Rule-based signals detected in the message: {signals}

Respond in this exact JSON format, with no extra text before or after:
{{
  "explanation": "2-3 simple sentences explaining why the message was likely classified this way",
  "suspicious_phrases": ["list", "of", "specific phrases or words from the message that look suspicious"],
  "safety_tips": ["2-3 short, practical safety tips relevant to this specific message"],
  "recommended_action": "one short sentence: e.g. 'Delete it', 'Verify with the sender directly', 'Safe to read'"
}}
"""


def _build_prompt(message: str, prediction: str, confidence: float, signals: dict) -> str:
    active_signals = [k.replace("_", " ") for k, v in signals.items() if v] or ["none detected"]
    return PROMPT_TEMPLATE.format(
        message=message.strip()[:1000],
        prediction=prediction.upper(),
        confidence=confidence,
        signals=", ".join(active_signals),
    )


def _call_groq(prompt: str) -> str:
    api_key = os.environ["GROQ_API_KEY"]
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_openai(prompt: str) -> str:
    api_key = os.environ["OPENAI_API_KEY"]
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(prompt: str) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


_PROVIDERS = {"groq": _call_groq, "openai": _call_openai, "gemini": _call_gemini}


def _parse_json_response(raw: str) -> dict:
    """LLMs sometimes wrap JSON in markdown fences -- strip those before parsing."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("json", "", 1)
    return json.loads(cleaned)


def explain_prediction(message: str, prediction: str, confidence: float, signals: dict) -> dict:
    """
    Calls the configured LLM provider and returns a dict with:
    explanation, suspicious_phrases, safety_tips, recommended_action.
    Falls back to a rule-based explanation if no API key is configured,
    so the app still works out of the box for demos.
    """
    prompt = _build_prompt(message, prediction, confidence, signals)

    call_fn = _PROVIDERS.get(PROVIDER)
    key_env_var = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}[PROVIDER]

    if call_fn is None or not os.environ.get(key_env_var):
        return _fallback_explanation(prediction, signals)

    try:
        raw = call_fn(prompt)
        return _parse_json_response(raw)
    except Exception as e:
        result = _fallback_explanation(prediction, signals)
        result["explanation"] += f" (Note: LLM call failed, showing rule-based explanation. Error: {e})"
        return result


def _fallback_explanation(prediction: str, signals: dict) -> dict:
    """Used when no API key is set, so the Streamlit app is still fully demoable."""
    active = [k.replace("_", " ") for k, v in signals.items() if v]
    if prediction == "spam":
        explanation = (
            "This message was flagged as spam because it shares patterns common in "
            "unsolicited or scam messages, such as urgency, unusual links, or promotional language."
        )
        tips = [
            "Do not click any links or call any numbers in the message.",
            "Never share personal or financial information in response to an unexpected message.",
            "Verify the sender through an official, separately-looked-up channel before acting.",
        ]
        action = "Do not respond -- delete or report it."
    else:
        explanation = "This message looks like normal personal or informational communication with no strong spam indicators."
        tips = ["Still avoid clicking unfamiliar links, even in messages that look safe.",
                "Confirm unexpected requests for money or info through another channel."]
        action = "Safe to read, but stay alert as usual."

    return {
        "explanation": explanation,
        "suspicious_phrases": active if active else ["No strong rule-based signals detected"],
        "safety_tips": tips,
        "recommended_action": action,
    }
