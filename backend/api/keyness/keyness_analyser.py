# api/keyness/keyness_analyser.py
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

nltk.download("punkt", quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not downloaded, download it
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def compute_keyness(uploaded_text, corpus_text, top_n=20):
    """
    Compute log-likelihood keyness between uploaded_text and corpus_text using NLTK.
    Returns top_n results as a list of dicts.
    """
    # Clean and tokenize
    def clean_tokens(text):
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if len(t) > 2]  # remove very short words
        return tokens

    tokens_a = clean_tokens(uploaded_text)
    tokens_b = clean_tokens(corpus_text)

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
            "uploaded_freq": a / len(tokens_a) * 1000,  # frequency per 1000 words
            "sample_freq": b / len(tokens_b) * 1000,  # frequency per 1000 words
            "count_a": int(a),
            "count_b": int(b),
            "effect_size": float(a - b) / max(1, len(tokens_a) + len(tokens_b)),  # optional
            "keyness": "Positive" if a > b else "Negative"
        })

    # Sort by log-likelihood descending and return top_n
    results.sort(key=lambda x: x["log_likelihood"], reverse=True)
    return results[:top_n]


def keyness_gensim(uploaded_text, corpus_text, top_n=20):
    import re
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel

    def clean_tokens(text):
        text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 2]
        return tokens

    # Tokenize uploaded text
    tokens_uploaded = clean_tokens(uploaded_text)

    # Split corpus into "documents" (one per line or sentence)
    corpus_docs = [clean_tokens(line) for line in corpus_text.splitlines() if line.strip()]
    if not corpus_docs:  # fallback if empty
        corpus_docs = [clean_tokens(corpus_text)]

    # Dictionary from both uploaded text and corpus documents
    dictionary = Dictionary([tokens_uploaded] + corpus_docs)

    # Create bag-of-words corpus
    bow_uploaded = dictionary.doc2bow(tokens_uploaded)
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus_docs]

    # TF-IDF model trained on all documents (uploaded + corpus)
    tfidf_model = TfidfModel([bow_uploaded] + bow_corpus)

    # Compute TF-IDF scores for uploaded text
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

def keyness_spacy(uploaded_text, corpus_text, top_n=20):
    def clean_tokens(text):
        doc = nlp(text.lower())
        tokens = [t.text for t in doc if t.is_alpha and len(t.text) > 2]  # remove punctuation & short words
        return tokens

    tokens_uploaded = clean_tokens(uploaded_text)
    tokens_corpus = clean_tokens(corpus_text)

    total_uploaded = len(tokens_uploaded)
    total_corpus = len(tokens_corpus)

    freq_uploaded = Counter(tokens_uploaded)
    freq_corpus = Counter(tokens_corpus)

    results = []
    for word in set(freq_uploaded.keys()).union(freq_corpus.keys()):
        a = freq_uploaded.get(word, 0)
        b = freq_corpus.get(word, 0)
        if a + b < 2:  # skip very rare words
            continue

        # contingency table for chi-square
        contingency = np.array([
            [a, total_uploaded - a],
            [b, total_corpus - b]
        ])

        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        # effect size (simple normalisation)
        effect_size = (a - b) / max(1, total_uploaded + total_corpus)

        results.append({
            "word": word,
            "uploaded_count": a,
            "sample_count": b,
            "chi2": float(chi2),
            "p_value": float(p),
            "log_likelihood": float(chi2),       # for scatter chart compatibility
            "effect_size": float(effect_size),   # for scatter chart
            "keyness": "Positive" if a > b else "Negative"
        })

    # Sort by chi2 descending and return top N
    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    return {
        "results": results_sorted[:top_n],
        "uploaded_total": total_uploaded,
        "corpus_total": total_corpus
    }