# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import math
import os

import corpus_toolkit as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models

app = Flask(__name__)
CORS(app)

# -----------------------------
# Sample corpus (reference)
# -----------------------------
# Path to your corpus file
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "sample1.txt")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    sample_corpus = f.read()

# -----------------------------
# Utility: preprocess text
# -----------------------------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stopwords.words("english")]

# -----------------------------
# NLTK Keyness (log-likelihood ratio)
# -----------------------------
def nltk_keyness(uploaded, reference):
    uploaded_tokens = preprocess(uploaded)
    ref_tokens = preprocess(reference)

    uploaded_counts = Counter(uploaded_tokens)
    ref_counts = Counter(ref_tokens)

    all_words = set(uploaded_counts.keys()).union(ref_counts.keys())
    results = []

    for word in all_words:
        O1 = uploaded_counts.get(word, 0)
        O2 = ref_counts.get(word, 0)
        N1 = sum(uploaded_counts.values())
        N2 = sum(ref_counts.values())

        # expected frequencies
        E1 = N1 * (O1 + O2) / (N1 + N2)
        E2 = N2 * (O1 + O2) / (N1 + N2)

        # log-likelihood ratio
        if O1 > 0 and E1 > 0:
            LL1 = O1 * math.log(O1 / E1)
        else:
            LL1 = 0
        if O2 > 0 and E2 > 0:
            LL2 = O2 * math.log(O2 / E2)
        else:
            LL2 = 0
        LL = 2 * (LL1 + LL2)

        results.append({"word": word, "score": LL})

    return sorted(results, key=lambda x: -x["score"])[:10]

# -----------------------------
# Corpus-Toolkit Keyness
# -----------------------------
def corpus_toolkit_keyness(uploaded, reference):
    # Make corpus-toolkit corpora
    ct.make_corpus("uploaded_corpus", uploaded)
    ct.make_corpus("reference_corpus", reference)

    keyness_result = ct.keyness("uploaded_corpus", "reference_corpus", "chi2")
    results = [{"word": k, "score": float(v)} for k, v in keyness_result.items()][:10]

    return results

# -----------------------------
# Sklearn TF-IDF
# -----------------------------
def sklearn_keyness(uploaded, reference):
    docs = [uploaded, reference]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(docs)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray()[0]  # first doc (uploaded_text)

    scores = [
        {"word": feature_names[i], "score": tfidf_scores[i]}
        for i in range(len(feature_names))
        if tfidf_scores[i] > 0
    ]
    return sorted(scores, key=lambda x: -x["score"])[:10]

# -----------------------------
# Gensim TF-IDF
# -----------------------------
def gensim_keyness(uploaded, reference):
    texts = [preprocess(uploaded), preprocess(reference)]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]

    tfidf_model = models.TfidfModel(corpus)
    tfidf_corpus = tfidf_model[corpus[0]]  # first doc (uploaded_text)

    results = [
        {"word": dictionary[w_id], "score": float(score)}
        for w_id, score in tfidf_corpus
    ]
    return sorted(results, key=lambda x: -x["score"])[:10]

# -----------------------------
# API Route
# -----------------------------
@app.route("/api/keyness/compare", methods=["POST"])
def compare_keyness():
    data = request.get_json()
    uploaded_text = data.get("text", "")

    if not uploaded_text.strip():
        return jsonify({"error": "No text provided"}), 400

    results = {
        "nltk": nltk_keyness(uploaded_text, sample_corpus),
        "corpus_toolkit": corpus_toolkit_keyness(uploaded_text, sample_corpus),
        "sklearn": sklearn_keyness(uploaded_text, sample_corpus),
        "gensim": gensim_keyness(uploaded_text, sample_corpus),
    }

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
