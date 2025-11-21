import streamlit as st
import joblib
import re
import os
import numpy as np

# ---------------------------
# Clean text function
# ---------------------------
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'<[^>]+>', ' ', s)       # remove HTML
    s = re.sub(r'http\S+', ' ', s)       # remove URLs
    s = re.sub(r'[^a-z0-9\s]', ' ', s)   # remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------------------
# Safe loading with checks
# ---------------------------
@st.cache_resource
def load_artifacts():

    tfidf_path = "tfidf_vectorizer.pkl"
    model_path = "logreg_model.pkl"

    if not os.path.exists(tfidf_path):
        st.error(f"❌ Missing file: {tfidf_path}. Upload it to your GitHub repo.")
        st.stop()

    if not os.path.exists(model_path):
        st.error(f"❌ Missing file: {model_path}. Upload it to your GitHub repo.")
        st.stop()

    try:
        tfidf = joblib.load(tfidf_path)
    except Exception as e:
        st.error(f"❌ Could not load TF-IDF vectorizer:\n\n{e}")
        st.stop()

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Could not load model:\n\n{e}")
        st.stop()

    return tfidf, model


tfidf, model = load_artifacts()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write("Paste any job posting text to detect if it's fake or real using ML classification.")

user_input = st.text_area("Enter job description here:", height=220)

if st.button("Analyze"):

    if len(user_input.strip()) == 0:
        st.warning("⚠ Please enter some text.")
        st.stop()

    # Clean text
    cleaned = clean_text(user_input)

    # Convert to vector
    try:
        vector = tfidf.transform([cleaned])
    except Exception as e:
        st.error(f"❌ Error while vectorizing:\n\n{e}")
        st.stop()

    # Validate feature count
    try:
        expected = model.n_features_in_
        got = vector.shape[1]

        if expected != got:
            st.error(
                f"❌ **Model / Vectorizer mismatch**\n\n"
                f"The loaded vectorizer has **{got} features**, but the model expects **{expected}**.\n\n"
                f"➡ You likely uploaded the wrong TF-IDF file or the wrong model file.\n"
                f"Please upload the *original pair* used during training."
            )
            st.stop()

    except Exception as e:
        st.error(f"❌ Could not validate feature count:\n\n{e}")
        st.stop()

    # Predict
    try:
        pred = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0][1]
    except Exception as e:
        st.error(f"❌ Prediction error:\n\n{e}")
        st.stop()

    # Output
    if pred == 1:
        st.error(f"🚨 Fake Job Posting Detected! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Real Job Posting (Fake Confidence: {proba:.2f})")

    st.caption("Confidence refers to the model's probability that the posting is fake.")
