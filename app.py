# -*- coding: utf-8 -*-
"""
Streamlit UI for the WOS-46985 Domain Classifier.
Only loads pre-trained artifacts produced by train.py — no training happens here.

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from nlp_utils import preprocess_text

MODEL_DIR = "models"
MIN_WORDS_FOR_RELIABLE_PREDICTION = 20


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit re-runs the whole script on every interaction,
# so @st.cache_resource makes sure the (large) model files are loaded once
# and reused, not re-read from disk on every keystroke.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.joblib"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    dt_model = joblib.load(os.path.join(MODEL_DIR, "dt_model.joblib"))
    lr_model = joblib.load(os.path.join(MODEL_DIR, "lr_model.joblib"))
    return tfidf, label_encoder, dt_model, lr_model


@st.cache_resource
def load_lstm_if_available():
    """LSTM is optional — app still works fine without it."""
    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    tok_path = os.path.join(MODEL_DIR, "abs_tokenizer.joblib")
    cfg_path = os.path.join(MODEL_DIR, "lstm_config.joblib")
    if not (os.path.exists(model_path) and os.path.exists(tok_path)):
        return None, None, None
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tok_path)
    config = joblib.load(cfg_path)
    return model, tokenizer, config


@st.cache_data
def load_comparison_table():
    path = os.path.join(MODEL_DIR, "model_comparison_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


tfidf, label_encoder, dt_model, lr_model = load_artifacts()
lstm_model, abs_tokenizer, lstm_config = load_lstm_if_available()

MODEL_MAP = {"Logistic Regression": lr_model, "Decision Tree": dt_model}
if lstm_model is not None:
    MODEL_MAP["BiLSTM"] = lstm_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(text: str, model_name: str, top_n: int = 3):
    word_count = len(text.split())
    if word_count < MIN_WORDS_FOR_RELIABLE_PREDICTION:
        return None, f"Input has only {word_count} word(s). Paste a fuller abstract (~20+ words) for a reliable prediction."

    cleaned = preprocess_text(text)

    if model_name == "BiLSTM":
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = abs_tokenizer.texts_to_sequences([cleaned])
        if seq[0] == []:
            return None, "None of the words matched the model's vocabulary."
        padded = pad_sequences(seq, maxlen=lstm_config["max_length"], padding="post", truncating="post")
        proba = lstm_model.predict(padded, verbose=0)[0]
    else:
        vec = tfidf.transform([cleaned])
        if vec.nnz == 0:
            return None, "None of the words matched the model's vocabulary (TF-IDF vector is all zeros)."
        model = MODEL_MAP[model_name]
        proba = model.predict_proba(vec)[0]

    top_idx = proba.argsort()[-top_n:][::-1]
    ranked = [(label_encoder.inverse_transform([i])[0], proba[i] * 100) for i in top_idx]
    return ranked, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Research Domain Classifier", layout="wide")
st.title("📄 Research Abstract Domain Classifier")
st.caption("WOS-46985 dataset — predicts one of 7 academic domains from an abstract.")

tab_predict, tab_compare = st.tabs(["🔮 Predict", "📊 Model Comparison"])

with tab_predict:
    col1, col2 = st.columns([2, 1])

    with col1:
        abstract_input = st.text_area(
            "Paste an abstract",
            height=220,
            placeholder="Paste a full research abstract here (~200 words works best)...",
        )
        keywords_input = st.text_input("Keywords (optional, semicolon-separated)", "")

    with col2:
        model_choice = st.radio("Model", list(MODEL_MAP.keys()))
        top_n = st.slider("Show top N predictions", 1, 5, 3)
        run = st.button("Predict Domain", type="primary", use_container_width=True)

    if run:
        if not abstract_input.strip():
            st.warning("Please paste an abstract first.")
        else:
            ranked, error = predict(abstract_input, model_choice, top_n=top_n)
            if error:
                st.warning(f"⚠️ {error}")
            else:
                best_label, best_conf = ranked[0]
                st.success(f"**Predicted Domain: {best_label}**  ({best_conf:.1f}% confidence)")
                chart_df = pd.DataFrame(ranked, columns=["Domain", "Confidence (%)"]).set_index("Domain")
                st.bar_chart(chart_df)

with tab_compare:
    st.subheader("Model performance on the held-out test set")
    df = load_comparison_table()
    if df is None:
        st.info("Run `train.py` first to generate model_comparison_results.csv.")
    else:
        st.dataframe(df, use_container_width=True)
        metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1"] if c in df.columns]
        st.bar_chart(df.set_index("Model")[metric_cols])
