import re
import math
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from collections import defaultdict
import numpy as np
from scipy.stats import chi2_contingency
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import spacy
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("punkt", quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Allowed POS tags (excluding PROPN / proper nouns)
ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}

def filter_content_words(text):
    """
    Tokenize with spaCy and keep only content words (NOUN, VERB, ADJ, ADV).
    Uses original casing for POS tagging to correctly identify proper nouns,
    but returns lemmatized lowercase tokens for analysis.
    """
    doc = nlp(text)  # ✅ keep original casing for POS
    return [
        token.lemma_.lower()  # return lowercase lemmatized token
        for token in doc
        if token.is_alpha and len(token) > 2 and token.pos_ in ALLOWED_POS
    ]

# ---------------- NLTK Keyness ----------------
def compute_keyness(uploaded_text, corpus_text, top_n=20):
    tokens_a = filter_content_words(uploaded_text)
    tokens_b = filter_content_words(corpus_text)

    freq_a = Counter(tokens_a)
    freq_b = Counter(tokens_b)

    total_a, total_b = len(tokens_a), len(tokens_b)

    results = []
    for word in set(freq_a.keys()).union(freq_b.keys()):
        a = freq_a.get(word, 0)
        b = freq_b.get(word, 0)
        if a + b < 2:
            continue
        c = total_a - a
        d = total_b - b
        try:
            e1 = c * (a + b) / (c + d)
            e2 = d * (a + b) / (c + d)
            ll = 2 * ((a * math.log(a / e1)) + (b * math.log(b / e2)))
        except (ValueError, ZeroDivisionError):
            ll = 0
        results.append({
            "word": word,
            "log_likelihood": float(ll),
            "uploaded_freq": a / len(tokens_a) * 1000 if tokens_a else 0,
            "sample_freq": b / len(tokens_b) * 1000 if tokens_b else 0,
            "count_a": int(a),
            "count_b": int(b),
            "effect_size": float(a - b) / max(1, len(tokens_a) + len(tokens_b)),
            "keyness": "Positive" if a > b else "Negative"
        })

    results.sort(key=lambda x: x["log_likelihood"], reverse=True)
    return results[:top_n]

# ---------------- Gensim Keyness ----------------
def keyness_gensim(uploaded_text, corpus_text, top_n=20):
    tokens_uploaded = filter_content_words(uploaded_text)
    corpus_docs = [filter_content_words(line) for line in corpus_text.splitlines() if line.strip()]
    if not corpus_docs:
        corpus_docs = [filter_content_words(corpus_text)]

    dictionary = Dictionary([tokens_uploaded] + corpus_docs)
    bow_uploaded = dictionary.doc2bow(tokens_uploaded)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_docs]

    tfidf_model = TfidfModel([bow_uploaded] + bow_corpus)
    tfidf_scores = tfidf_model[bow_uploaded]

    results = []
    for word_id, score in tfidf_scores:
        word = dictionary[word_id]
        count_uploaded = tokens_uploaded.count(word)
        count_corpus = sum(doc.count(word) for doc in corpus_docs)
        results.append({
            "word": word,
            "uploaded_count": count_uploaded,
            "sample_count": count_corpus,
            "tfidf_score": float(score)
        })

    results_sorted = sorted(results, key=lambda x: x["tfidf_score"], reverse=True)
    return {
        "results": results_sorted[:top_n],
        "uploaded_total": len(tokens_uploaded),
        "corpus_total": sum(len(doc) for doc in corpus_docs)
    }

# ---------------- spaCy Keyness ----------------
def keyness_spacy(uploaded_text, corpus_text, top_n=20):
    tokens_uploaded = filter_content_words(uploaded_text)
    tokens_corpus = filter_content_words(corpus_text)

    total_uploaded = len(tokens_uploaded)
    total_corpus = len(tokens_corpus)

    freq_uploaded = Counter(tokens_uploaded)
    freq_corpus = Counter(tokens_corpus)

    results = []
    for word in set(freq_uploaded.keys()).union(freq_corpus.keys()):
        a = freq_uploaded.get(word, 0)
        b = freq_corpus.get(word, 0)
        if a + b < 2:
            continue

        contingency = np.array([
            [a, total_uploaded - a],
            [b, total_corpus - b]
        ])

        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        effect_size = (a - b) / max(1, total_uploaded + total_corpus)

        results.append({
            "word": word,
            "uploaded_count": a,
            "sample_count": b,
            "chi2": float(chi2),
            "p_value": float(p),
            "log_likelihood": float(chi2),
            "effect_size": float(effect_size),
            "keyness": "Positive" if a > b else "Negative"
        })

    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    return {
        "results": results_sorted[:top_n],
        "uploaded_total": total_uploaded,
        "corpus_total": total_corpus
    }

# ---------------- scikit-learn Keyness ----------------
def keyness_sklearn(uploaded_text, corpus_text, top_n=20):
    vectorizer = CountVectorizer(tokenizer=filter_content_words, lowercase=True)
    X = vectorizer.fit_transform([uploaded_text, corpus_text])
    terms = vectorizer.get_feature_names_out()
    freqs = X.toarray()

    uploaded_counts = freqs[0]
    corpus_counts = freqs[1]

    results = []
    for word, u_count, c_count in zip(terms, uploaded_counts, corpus_counts):
        if u_count == 0:
            continue

        contingency = np.array([
            [u_count, sum(uploaded_counts) - u_count],
            [c_count, sum(corpus_counts) - c_count]
        ])

        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        results.append({
            "word": word,
            "uploaded_count": int(u_count),
            "sample_count": int(c_count),
            "chi2": float(chi2),
            "p_value": float(p),
        })

    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    results_top = results_sorted[:top_n]

    return {
        "results": results_top,
        "uploaded_total": int(sum(uploaded_counts)),
        "corpus_total": int(sum(corpus_counts)),
    }
