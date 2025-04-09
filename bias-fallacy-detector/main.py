import pandas as pd
import string
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------------
# Step 1: Load Hugging Face Dataset
# -------------------------
print("📦 Loading bias detection dataset from Hugging Face...")
hf_data = load_dataset("pranjali97/Bias-detection-combined")
df = hf_data['train'].to_pandas()

# -------------------------
# Step 2: Preprocess
# -------------------------
df = df[['text', 'label']].dropna()
df = df.rename(columns={'text': 'sentence', 'label': 'Label_bias'})

# Show class distribution
print("📊 Class distribution:")
print(df['Label_bias'].value_counts())

# Clean the sentence text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['clean_text'] = df['sentence'].apply(clean_text)

# -------------------------
# Step 3: Train/test split and vectorization
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['Label_bias'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------
# Step 4: Train model
# -------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -------------------------
# Step 5: Evaluate
# -------------------------
y_pred = model.predict(X_test_vec)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# Step 6: Save model and vectorizer
# -------------------------
joblib.dump(model, "models/bias_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("✅ Model and vectorizer saved to /models/")
