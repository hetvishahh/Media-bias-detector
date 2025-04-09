import streamlit as st
import joblib
import string
import numpy as np
import pandas as pd
from datetime import datetime
from newspaper import Article

st.set_page_config(page_title="Media Bias Detector", page_icon="🧠")

# ----------------------------
# Load Model + Vectorizer
# ----------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("models/bias_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ----------------------------
# Clean Text
# ----------------------------
def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

# ----------------------------
# Highlight Important Words
# ----------------------------
def highlight_words(text, vectorizer, model):
    cleaned = clean_text(text)
    words = cleaned.split()
    tfidf = vectorizer.transform([cleaned])
    weights = model.coef_[0]

    highlighted = []
    for word in words:
        if word in vectorizer.vocabulary_:
            index = vectorizer.vocabulary_[word]
            score = weights[index]
            if score > 0:
                highlighted.append(f"**:red[{word}]**")
            elif score < 0:
                highlighted.append(f"**:green[{word}]**")
            else:
                highlighted.append(word)
        else:
            highlighted.append(word)
    return " ".join(highlighted)

# ----------------------------
# Log Feedback
# ----------------------------
def log_feedback(text, prediction, feedback):
    df = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "model_prediction": prediction,
        "user_feedback": feedback
    }])

    # Print confirmation (for debugging)
    print("Feedback received:", feedback)

    # Save the feedback to CSV
    df.to_csv("feedback.csv", mode='a', index=False, header=not pd.io.common.file_exists("feedback.csv"))

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 Media Bias Detector")
st.markdown("Analyze bias in your own text or in articles from the web using a lightweight ML model.")
st.markdown("This tool is not 100% accurate. This is simply for educational purposes.")
# ----------------------------
# User Text Input
# ----------------------------
st.header("📝 Analyze Your Own Text")
user_input = st.text_area("Paste a sentence or paragraph:")

if st.button("Analyze Text"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec)[0][prediction]
        label = "🟥 BIASED" if prediction == 1 else "🟩 NEUTRAL"

        st.markdown(f"### Result: {label} (Confidence: {confidence:.2%})")
        st.markdown("### 🔍 Highlighted Words")
        st.markdown(highlight_words(user_input, vectorizer, model))

        # Feedback section with debugging
        st.markdown("### 🙋 Was this correct?")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("👍 Yes"):
                print("Feedback: Correct")  # Debug line
                log_feedback(user_input, prediction, "Correct")
                st.success("✅ Thanks for your feedback!")
        with col2:
            if st.button("👎 No"):
                print("Feedback: Incorrect")  # Debug line
                log_feedback(user_input, prediction, "Incorrect")
                st.success("✅ Feedback saved!")
        with col3:
            if st.button("🤔 Not Sure/Maybe"):
                print("Feedback: Not Sure")  # Debug line
                log_feedback(user_input, prediction, "Not Sure")
                st.success("✅ Noted.")


# ----------------------------
# Analyze from Article URL
# ----------------------------
st.markdown("---")
st.header("🔗 Analyze a News Article URL")
url_input = st.text_input("Paste a news article URL:")

if st.button("Analyze Article"):
    try:
        article = Article(url_input)
        article.download()
        article.parse()
        article_text = article.text.strip()

        if not article_text:
            raise ValueError("Article text is empty. The website may block scrapers.")

        st.success("✅ Article loaded!")
        st.text_area("📰 Extracted Text", article_text[:1500], height=200)

        cleaned = clean_text(article_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec)[0][prediction]
        label = "🟥 BIASED" if prediction == 1 else "🟩 NEUTRAL"

        st.markdown(f"### Result: {label} (Confidence: {confidence:.2%})")
        st.markdown("### 🔍 Highlighted Words")
        st.markdown(highlight_words(article_text, vectorizer, model))

        # Feedback
        st.markdown("### 🙋 Was this accurate?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Yes (URL)"):
                log_feedback(article_text, prediction, "Correct")
                st.success("✅ Feedback saved!")
        with col2:
            if st.button("👎 No (URL)"):
                log_feedback(article_text, prediction, "Incorrect")
                st.success("✅ Feedback saved!")
        with col3:
            if st.button("🤔 Not Sure (URL)"):
                log_feedback(article_text, prediction, "Not Sure")
                st.success("✅ Got it!")

    except Exception as e:
        st.error(f"❌ Could not extract article: {e}")
