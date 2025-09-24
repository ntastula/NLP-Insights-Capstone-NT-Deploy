import re
import math
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim import corpora, models
import numpy as np
from scipy.stats import chi2_contingency
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag, FreqDist
from math import log

nltk.download("punkt", quiet=True)

lemmatizer = WordNetLemmatizer()

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
    """
    Keep only nouns, verbs, adjectives, adverbs.
    Also splits concatenated words if needed.
    """
    # Tokenize sentences first
    tokens = word_tokenize(text)

    # Example: use spaCy or NLTK POS tagging
    # Here we use NLTK's pos_tag
    tagged = nltk.pos_tag(tokens)  # returns list of (word, pos)

    pos_map = {
        'NN': 'NOUN',
        'NNS': 'NOUN',
        # 'NNP': 'NOUN',
        # 'NNPS': 'NOUN',
        'VB': 'VERB',
        'VBD': 'VERB',
        'VBG': 'VERB',
        'VBN': 'VERB',
        'VBP': 'VERB',
        'VBZ': 'VERB',
        'JJ': 'ADJ',
        'JJR': 'ADJ',
        'JJS': 'ADJ',
        'RB': 'ADV',
        'RBR': 'ADV',
        'RBS': 'ADV',
    }

    filtered = []
    for word, pos in tagged:
        # Only alphabetic words longer than 2 chars
        if word.isalpha() and len(word) > 2:
            mapped_pos = pos_map.get(pos, "OTHER")
            if mapped_pos in ALLOWED_POS:
                # Split concatenated words using simple regex for camelCase or multiple lowercase
                split_words = re.findall(r'[A-Z]?[a-z]+', word)
                for w in split_words:
                    filtered.append({"word": w.lower(), "pos": mapped_pos})

    return filtered


def filter_all_words(text):
    """
    Keep all alphabetic tokens > 2 chars, lemmatized + lowercased.
    Also returns POS so that words can be coloured and grouped.
    """
    doc = nlp(text)
    filtered = []

    POS_MAP = {
        "NOUN": "NOUN",
        "PROPN": "NOUN",
        "VERB": "VERB",
        "ADJ": "ADJ",
        "ADV": "ADV",
    }

    for token in doc:
        if token.is_alpha and len(token) > 2:
            pos = POS_MAP.get(token.pos_, "OTHER")
            filtered.append({"word": token.lemma_.lower(), "pos": pos})

    return filtered


def extract_sentences(text, word):
    """
    Return all sentences from text containing the exact word (case-insensitive),
    ignoring punctuation attached to words.
    """
    if not text or not word:
        return []

    word_lower = word.lower()
    sentences = sent_tokenize(text)

    matched = []
    for s in sentences:
        # Use regex to match whole word boundaries
        if re.search(rf'\b{re.escape(word_lower)}\b', s, flags=re.IGNORECASE):
            matched.append(s)
    return matched

# ---------------------------
# Keyness Functions
# ---------------------------

def compute_keyness(uploaded_text, corpus_counts_map, top_n=50, filter_func=filter_content_words):
    """
    Compute a simple, writer-friendly keyness score:
    - Based on chi-square (like sklearn/gensim versions).
    - Groups results by POS (NOUN, VERB, ADJ, ADV).
    """

    if filter_func is None:
        filter_func = lambda text: [{"word": w.lower(), "pos": "OTHER"} for w in text.split()]

    # Tokenise uploaded text
    tokens_uploaded = filter_func(uploaded_text)
    uploaded_counts = Counter([t["word"] for t in tokens_uploaded])
    uploaded_total = sum(uploaded_counts.values())

    # Use provided corpus counts
    corpus_counts = corpus_counts_map.copy()
    corpus_total = sum(corpus_counts.values())

    results = []
    for word, u_count in uploaded_counts.items():
        c_count = corpus_counts.get(word, 0)

        contingency = np.array([
            [u_count, uploaded_total - u_count],
            [c_count, corpus_total - c_count]
        ])

        try:
            chi2_val, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2_val, p = 0, 1

        # POS tag (default to OTHER if not in ALLOWED_POS)
        pos = next((t["pos"] for t in tokens_uploaded if t["word"] == word), "OTHER")
        if pos not in ALLOWED_POS:
            pos = "OTHER"

        results.append({
            "word": word,
            "uploaded_count": u_count,
            "sample_count": c_count,
            "chi2": float(chi2_val),
            "p_value": float(p),
            "keyness": float(chi2_val),  # numeric score, not positive/negative
            "pos": pos
        })

    # Sort by chi2 descending
    sorted_results = sorted(results, key=lambda x: x["chi2"], reverse=True)

    return {
        "results": sorted_results[:top_n],
        "total_significant": len(sorted_results)
    }


def keyness_nltk(uploaded_text, corpus_counts_map, top_n=50, filter_func=filter_content_words):
    """NLTK-style log-likelihood using counts-based corpus."""
    if filter_func is None:
        filter_func = lambda t: [{"word": w.lower(), "pos": "OTHER"} for w in t.split()]

    uploaded_tokens = filter_func(uploaded_text)
    uploaded_counts = Counter([t["word"] for t in uploaded_tokens])

    uploaded_total = sum(uploaded_counts.values())
    corpus_total = sum(corpus_counts_map.values())

    results = []
    all_words = set(uploaded_counts.keys()).union(corpus_counts_map.keys())

    for word in all_words:
        count_a = uploaded_counts.get(word, 0)
        count_b = corpus_counts_map.get(word, 0)
        if count_a == 0:
            continue

        # Log-likelihood
        E1 = uploaded_total * (count_a + count_b) / (uploaded_total + corpus_total)
        E2 = corpus_total * (count_a + count_b) / (uploaded_total + corpus_total)
        ll = 2 * ((count_a * log(count_a / E1) if count_a > 0 else 0) +
                  (count_b * log(count_b / E2) if count_b > 0 else 0))

        effect_size = (count_a / uploaded_total - count_b / max(1, corpus_total))
        pos = next((t["pos"] for t in uploaded_tokens if t["word"] == word), "OTHER")
        if pos not in ALLOWED_POS:
            pos = "OTHER"

        results.append({
            "word": word,
            "uploaded_count": count_a,
            "sample_count": count_b,
            "log_likelihood": round(ll, 2),
            "effect_size": round(effect_size, 4),
            "keyness_score": round(ll, 2),  # numeric
            "direction": "Positive" if count_a > count_b else "Negative",
            "pos": pos
        })

    sorted_results = sorted(results, key=lambda x: x["keyness_score"], reverse=True)
    return {"results": sorted_results[:top_n], "total_significant": len(sorted_results)}



def keyness_gensim(uploaded_text, corpus_counts_map, top_n=50, filter_func=filter_content_words):
    """Gensim-style TF-IDF + chi2 keyness using counts-based corpus from _keyness.json"""
    if filter_func is None:
        filter_func = lambda t: [{"word": w.lower(), "pos": "OTHER"} for t in t.split()]

    # Tokens
    tokens_uploaded = filter_func(uploaded_text)
    words_uploaded = [t["word"] for t in tokens_uploaded]

    # Reconstruct "corpus text" from counts map
    words_corpus = []
    for word, count in corpus_counts_map.items():
        words_corpus.extend([word] * count)

    # Gensim dictionary & TF-IDF
    dictionary = corpora.Dictionary([words_uploaded, words_corpus])
    corpus_gensim = [dictionary.doc2bow(words_uploaded), dictionary.doc2bow(words_corpus)]
    tfidf = models.TfidfModel(corpus_gensim, smartirs="ntc")
    tfidf_scores = [tfidf[doc] for doc in corpus_gensim]

    uploaded_tfidf = {dictionary[id]: score for id, score in tfidf_scores[0]}

    uploaded_counts = Counter(words_uploaded)
    corpus_counts = Counter(words_corpus)

    uploaded_total = sum(uploaded_counts.values())
    corpus_total = sum(corpus_counts.values())

    results = []
    for word in uploaded_counts.keys():
        u_count = uploaded_counts[word]
        c_count = corpus_counts.get(word, 0)

        contingency = np.array([
            [u_count, uploaded_total - u_count],
            [c_count, corpus_total - c_count]
        ])
        try:
            chi2_val, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2_val, p = 0, 1

        pos = next((t["pos"] for t in tokens_uploaded if t["word"] == word), "OTHER")
        if pos not in ALLOWED_POS:
            pos = "OTHER"

        results.append({
            "word": word,
            "uploaded_count": u_count,
            "sample_count": c_count,
            "chi2": float(chi2_val),
            "p_value": float(p),
            "tfidf_score": float(uploaded_tfidf.get(word, 0)),
            "keyness_score": float(chi2_val),  # numeric
            "direction": "Positive" if u_count > c_count else "Negative",
            "pos": pos
        })

    results_sorted = sorted(results, key=lambda x: x["keyness_score"], reverse=True)
    return {
        "results": results_sorted[:top_n],
        "total_significant": len(results_sorted),
        "uploaded_total": uploaded_total,
        "corpus_total": corpus_total
    }



def keyness_spacy(uploaded_text, corpus_counts_map, top_n=50, filter_func=filter_content_words):
    if filter_func is None:
        filter_func = lambda t: [{"word": w.lower(), "pos": "OTHER"} for t in t.split()]

    tokens_uploaded = filter_func(uploaded_text)
    uploaded_counts = Counter([t["word"] for t in tokens_uploaded])
    corpus_counts = corpus_counts_map.copy()

    uploaded_total = sum(uploaded_counts.values())
    corpus_total = sum(corpus_counts.values())

    results = []
    for word, a in uploaded_counts.items():
        b = corpus_counts.get(word, 0)
        contingency = np.array([
            [a, uploaded_total - a],
            [b, corpus_total - b]
        ])
        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        effect_size = (a - b) / max(1, uploaded_total + corpus_total)
        pos = next((t["pos"] for t in tokens_uploaded if t["word"] == word), "OTHER")
        if pos not in ALLOWED_POS:
            pos = "OTHER"

        results.append({
            "word": word,
            "uploaded_count": a,
            "sample_count": b,
            "chi2": float(chi2),
            "p_value": float(p),
            "log_likelihood": float(chi2),  # keep for compatibility
            "effect_size": float(effect_size),
            "keyness_score": float(chi2),   # numeric, consistent across methods
            "direction": "Positive" if a > b else "Negative",  # keep for filtering
            "pos": pos
        })

    sorted_results = sorted(results, key=lambda x: x["chi2"], reverse=True)
    return {"results": sorted_results[:top_n], "total_significant": len(sorted_results)}



def keyness_sklearn(uploaded_text, corpus_counts_map, top_n=50, filter_func=None):
    if filter_func is None:
        filter_func = lambda t: [{"word": w.lower(), "pos": "OTHER"} for t in t.split()]

    tokens_uploaded = filter_func(uploaded_text)
    uploaded_counts = Counter([t["word"] for t in tokens_uploaded])
    corpus_counts = corpus_counts_map.copy()

    uploaded_total = sum(uploaded_counts.values())
    corpus_total = sum(corpus_counts.values())

    results = []
    for word, u_count in uploaded_counts.items():
        c_count = corpus_counts.get(word, 0)
        contingency = np.array([
            [u_count, uploaded_total - u_count],
            [c_count, corpus_total - c_count]
        ])
        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        pos = next((t["pos"] for t in tokens_uploaded if t["word"] == word), "OTHER")
        if pos not in ALLOWED_POS:
            pos = "OTHER"

        results.append({
            "word": word,
            "uploaded_count": u_count,
            "sample_count": c_count,
            "chi2": float(chi2),
            "p_value": float(p),
            "keyness_score": float(chi2),  # numeric
            "direction": "Positive" if u_count > c_count else "Negative",
            "pos": pos
        })

    sorted_results = sorted(results, key=lambda x: x["keyness_score"], reverse=True)
    return {"results": sorted_results[:top_n], "total_significant": len(sorted_results)}

