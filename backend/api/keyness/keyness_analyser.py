import re
import math
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from gensim import corpora, models
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

ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# ---------------------------
# Filtering functions
# ---------------------------

def filter_content_words(text):
    """Keep only nouns, verbs, adjectives, and adverbs."""
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and len(token) > 2 and token.pos_ in ALLOWED_POS
    ]


def filter_all_words(text):
    """Keep all alphabetic tokens > 2 chars, lemmatized + lowercased."""
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and len(token) > 2
    ]


# ---------------------------
# Keyness Functions
# ---------------------------

def compute_keyness(uploaded_text, corpus_text, top_n=20, filter_func=filter_content_words):
    tokens_a = filter_func(uploaded_text)
    tokens_b = filter_func(corpus_text)

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
            "uploaded_freq": a / total_a * 1000 if total_a else 0,
            "sample_freq": b / total_b * 1000 if total_b else 0,
            "count_a": int(a),
            "count_b": int(b),
            "effect_size": float(a - b) / max(1, total_a + total_b),
            "keyness": "Positive" if a > b else "Negative"
        })

    results.sort(key=lambda x: x["log_likelihood"], reverse=True)
    return results[:top_n]


def keyness_gensim(uploaded_text, corpus_text, top_n=20, filter_func=filter_content_words):
    tokens_uploaded = filter_func(uploaded_text)
    corpus_tokens = filter_func(corpus_text)

    dictionary = corpora.Dictionary([tokens_uploaded, corpus_tokens])
    corpus_gensim = [dictionary.doc2bow(tokens_uploaded), dictionary.doc2bow(corpus_tokens)]

    tfidf = models.TfidfModel(corpus_gensim, smartirs='ntc')
    tfidf_scores = [tfidf[doc] for doc in corpus_gensim]

    uploaded_tfidf = {dictionary[id]: score for id, score in tfidf_scores[0]}
    corpus_tfidf = {dictionary[id]: score for id, score in tfidf_scores[1]}

    uploaded_counts = Counter(tokens_uploaded)
    corpus_counts = Counter(corpus_tokens)

    results = []
    for word in set(tokens_uploaded).union(corpus_tokens):
        u_count = uploaded_counts[word]
        c_count = corpus_counts.get(word, 0)

        contingency = np.array([
            [u_count, sum(uploaded_counts.values()) - u_count],
            [c_count, sum(corpus_counts.values()) - c_count]
        ])
        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        results.append({
            "word": word,
            "uploaded_count": u_count,
            "sample_count": c_count,
            "chi2": float(chi2),
            "p_value": float(p),
            "tfidf_score": float(uploaded_tfidf.get(word, 0))
        })

    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    return {
        "results": results_sorted[:top_n],
        "uploaded_total": len(tokens_uploaded),
        "corpus_total": len(corpus_tokens),

    }


def keyness_spacy(uploaded_text, corpus_text, top_n=20, filter_func=filter_content_words):
    tokens_uploaded = filter_func(uploaded_text)
    tokens_corpus = filter_func(corpus_text)

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
        "corpus_total": total_corpus,
    }


def keyness_sklearn(uploaded_text, corpus_text, top_n=20, filter_func=filter_content_words):
    tokens_uploaded = filter_func(uploaded_text)
    tokens_corpus = filter_func(corpus_text)

    vectorizer = CountVectorizer(vocabulary=list(set(tokens_uploaded + tokens_corpus)))
    X = vectorizer.fit_transform([" ".join(tokens_uploaded), " ".join(tokens_corpus)])
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
    return {
        "results": results_sorted[:top_n],
        "uploaded_total": len(tokens_uploaded),
        "corpus_total": len(tokens_corpus),

    }
