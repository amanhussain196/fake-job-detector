import streamlit as st
import joblib
import re

# ---------------------------
# Text cleaning function
# ---------------------------
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'<[^>]+>', ' ', s)       # remove html
    s = re.sub(r'http\S+', ' ', s)       # remove urls
    s = re.sub(r'[^a-z0-9\s]', ' ', s)   # remove punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------------------
# Load model + vectorizer
# ---------------------------
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logreg_model.pkl")
    return tfidf, model

tfidf, model = load_artifacts()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🔍")

st.title("🔍 Fake Job Posting Detection")
st.write("Paste any job posting below and the model will classify it as **Real** or **Fake** based on NLP analysis.")

user_input = st.text_area("Enter Job Description:", height=250)

if st.button("Analyze"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0][1]  # probability of FAKE

        if prediction == 1:
            st.error(f"🚨 **Fake Job Posting Detected!** (Confidence: {proba:.2f})")
        else:
            st.success(f"✅ **Real Job Posting** (Confidence Fake: {proba:.2f})")

        # For transparency
        st.caption("Confidence refers to the model's probability that the post is fraudulent.")
