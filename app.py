import streamlit as st
import joblib
import re
import string
import os

# ===============================
# LOAD MODEL
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "resume_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "resume_vectorizer.pkl"))

# ===============================
# SIMPLE STOPWORDS
# ===============================

stop_words = {
    "and", "or", "the", "a", "an", "in", "on", "with",
    "for", "to", "of", "is", "are", "was", "were",
    "this", "that", "i", "have", "has", "had",
    "be", "been", "being", "at", "by", "as",
    "from", "it", "its", "my", "our", "your"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="AI Resume Screening", layout="centered")

st.title("🤖 AI Resume Screening System")
st.write("### Position: Data Analyst")

resume_input = st.text_area("📄 Paste Resume Text Here:", height=200)

if st.button("🔍 Screen Resume"):

    if resume_input.strip() == "":
        st.warning("Please enter resume text first.")
    else:
        cleaned = clean_text(resume_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        st.markdown("---")
        st.subheader("📊 Screening Result")

        if prediction == "Shortlisted":
            st.success("✅ Candidate Shortlisted")
        else:
            st.error("❌ Not Shortlisted")

        try:
            prob = model.predict_proba(vectorized)[0]
            confidence = max(prob) * 100
            st.info(f"Confidence Score: {confidence:.2f}%")
        except:
            pass

st.markdown("---")
st.caption("Built with NLP + TF-IDF + Logistic Regression")