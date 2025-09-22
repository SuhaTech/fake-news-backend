from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, traceback, pandas as pd, string, os
from scipy.sparse import hstack, csr_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, gdown

# ----------------------------
# Safe NLTK setup
# ----------------------------
def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

safe_nltk_download("tokenizers/punkt")
safe_nltk_download("corpora/stopwords")
safe_nltk_download("corpora/wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Text helpers
# ----------------------------
def remove_punct(text):
    return "".join([ch for ch in text if ch not in string.punctuation])

def remove_stopwords(words):
    return [w for w in words if w.lower() not in stop_words]

def lemmatizing(words):
    return [lemmatizer.lemmatize(w) for w in words]

# ----------------------------
# Google Drive Model Download
# ----------------------------
FILES = {
    "fake_news_model.pkl": "https://drive.google.com/uc?id=1FwTgjUBe4BKXkgJzYlDDf5YYXYn6B6Qx",
    "scaler.pkl": "https://drive.google.com/uc?id=163IvEU_KvBtS-xeiY0Hl7x6zu6NRfdCE",
    "text_vectorizer.pkl": "https://drive.google.com/uc?id=1gfoa4ZmOi0CK7DKHwiTaU0sExfBLjIew",
    "title_vectorizer.pkl": "https://drive.google.com/uc?id=1w-Spt6vt0qGy1CfRyXKKRSt8NkxNbrVE",
}

for filename, url in FILES.items():
    if not os.path.exists(filename):
        print(f"⬇️ Downloading {filename}...")
        gdown.download(url, filename, quiet=False)

# ----------------------------
# App init & CORS
# ----------------------------
app = Flask(__name__)
CORS(app)  # Allow all origins for now

# ----------------------------
# Load artifacts
# ----------------------------
try:
    text_vectorizer = pickle.load(open("text_vectorizer.pkl", "rb"))
    title_vectorizer = pickle.load(open("title_vectorizer.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    print("✅ Artifacts loaded. Classes:", getattr(model, "classes_", None))
except Exception as e:
    print("❌ Loading error:", e)
    traceback.print_exc()
    text_vectorizer = title_vectorizer = scaler = model = None

# ----------------------------
# Numeric features
# ----------------------------
def preprocess_features(title, text):
    chars_before = len(text.replace(" ", ""))
    words_cleaned = lemmatizing(remove_stopwords(word_tokenize(remove_punct(text))))
    chars_after = sum(len(word) for word in words_cleaned)
    text_uc_count = sum(1 for w in text.split() if w.isupper())
    title_uc_count = sum(1 for w in title.split() if w.isupper())
    return pd.DataFrame(
        [[chars_before, chars_after, text_uc_count, title_uc_count]],
        columns=["text_chars_before", "text_chars_after", "text_uc_count", "title_uc_count"]
    )

# ----------------------------
# Predict endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model or not text_vectorizer or not title_vectorizer or not scaler:
            return jsonify({"error": "Model/vectorizers/scaler not loaded"}), 500

        data = request.get_json(force=True)
        text = str(data.get("text", ""))
        title = str(data.get("title", ""))

        if not text and not title:
            return jsonify({"error": "No text or title provided"}), 400

        numeric_df = preprocess_features(title, text)
        numeric_sparse = csr_matrix(scaler.transform(numeric_df))

        features = hstack([
            title_vectorizer.transform([title]),
            text_vectorizer.transform([text]),
            numeric_sparse
        ])

        raw_pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 2)

        is_fake = True if str(raw_pred) == "Fake" else False
        label = "Fake" if is_fake else "Real"

        return jsonify({
            "isFake": is_fake,
            "label": label,
            "confidence": confidence,
            "raw_prediction": str(raw_pred)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Render-ready
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
