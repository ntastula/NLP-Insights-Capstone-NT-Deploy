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

def compute_keyness(uploaded_text, corpus_text=None, top_n=50, filter_func=None):
    """
    TEMP: Return top words with POS from uploaded_text only.
    Ignore corpus and keyness scoring.
    """
    if filter_func is None:
        # Simple fallback: split words and mark as OTHER
        filter_func = lambda text: [{"word": w.lower(), "pos": "OTHER"} for w in text.split()]

    tokens = filter_func(uploaded_text)

    # Count frequencies
    freq = Counter(t["word"] for t in tokens)

    # Get top_n words
    top_words = freq.most_common(top_n)

    # Build results with POS
    results = []
    for word, count in top_words:
        pos = next((t["pos"] for t in tokens if t["word"] == word), "OTHER")
        results.append({
            "word": word,
            "count_a": count,
            "count_b": 0,
            "log_likelihood": 0,  # placeholder
            "effect_size": 0,  # placeholder
            "keyness": "Positive",  # placeholder
            "pos": pos
        })

    # Group by POS
    pos_groups = {}
    for item in results:
        pos_groups.setdefault(item["pos"], []).append(item)

    return pos_groups

import nltk
from nltk import word_tokenize, pos_tag, FreqDist
from math import log

ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}

def keyness_nltk(uploaded_text, corpus_text, top_n=20, filter_func=None):
    """
    Compute keyness using NLTK-style log-likelihood.
    Returns a list of dicts with word, pos, counts, effect_size, log_likelihood, keyness.
    """

    # Optional filter function
    if filter_func:
        uploaded_tokens = filter_func(uploaded_text)
        corpus_tokens = filter_func(corpus_text)
        uploaded_words = [t["word"] for t in uploaded_tokens]
        corpus_words = [t["word"] for t in corpus_tokens]
        pos_map_input = {t["word"]: t["pos"] for t in uploaded_tokens + corpus_tokens}
    else:
        uploaded_words = word_tokenize(uploaded_text.lower())
        corpus_words = word_tokenize(corpus_text.lower())
        pos_map_input = {}

    # Frequency distributions
    freq_uploaded = FreqDist(uploaded_words)
    freq_corpus = FreqDist(corpus_words)

    total_uploaded = sum(freq_uploaded.values())
    total_corpus = sum(freq_corpus.values())

    results = []

    # Merge keys
    all_words = set(freq_uploaded.keys()).union(freq_corpus.keys())

    for word in all_words:
        count_a = freq_uploaded.get(word, 0)
        count_b = freq_corpus.get(word, 0)

        # Skip words not in uploaded_text (if you want only uploaded words in interactive list)
        if count_a == 0:
            continue

        # Compute log-likelihood
        E1 = total_uploaded * (count_a + count_b) / (total_uploaded + total_corpus)
        E2 = total_corpus * (count_a + count_b) / (total_uploaded + total_corpus)
        ll = 2 * (
            (count_a * log(count_a / E1) if count_a > 0 else 0) +
            (count_b * log(count_b / E2) if count_b > 0 else 0)
        )

        # Effect size (simple)
        effect_size = (count_a / total_uploaded) - (count_b / total_corpus)

        # Determine POS
        pos = pos_map_input.get(word)
        if not pos:
            # fallback: use NLTK pos_tag
            pos_tagged = pos_tag([word])
            pos_nltk = pos_tagged[0][1]
            pos_map = {
                'NN': 'NOUN', 'NNS': 'NOUN',
                'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
                'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
                'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
                'NNP': None, 'NNPS': None  # ignore proper nouns
            }
            pos = pos_map.get(pos_nltk, "OTHER")

        if pos not in ALLOWED_POS:
            pos = "OTHER"

        results.append({
            "word": word,
            "pos": pos,
            "uploaded_count": count_a,
            "sample_count": count_b,
            "effect_size": round(effect_size, 4),
            "log_likelihood": round(ll, 2),
            "keyness": round(ll, 2)
        })

    # Sort by log-likelihood descending
    results_list = sorted(results, key=lambda x: x["keyness"], reverse=True)[:top_n]

    return results_list



def keyness_gensim(uploaded_text, corpus_text, top_n=20, filter_func=filter_content_words):
    tokens_uploaded = filter_func(uploaded_text)
    tokens_corpus = filter_func(corpus_text)

    words_uploaded = [t['word'] if isinstance(t, dict) else t for t in tokens_uploaded]
    words_corpus = [t['word'] if isinstance(t, dict) else t for t in tokens_corpus]

    dictionary = corpora.Dictionary([words_uploaded, words_corpus])
    corpus_gensim = [dictionary.doc2bow(words_uploaded), dictionary.doc2bow(words_corpus)]

    tfidf = models.TfidfModel(corpus_gensim, smartirs='ntc')
    tfidf_scores = [tfidf[doc] for doc in corpus_gensim]

    uploaded_tfidf = {dictionary[id]: score for id, score in tfidf_scores[0]}
    corpus_tfidf = {dictionary[id]: score for id, score in tfidf_scores[1]}

    uploaded_counts = Counter(words_uploaded)
    corpus_counts = Counter(words_corpus)

    results = []
    for word in uploaded_counts.keys():
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

        pos = None
        for t in tokens_uploaded:
            if isinstance(t, dict) and t['word'] == word:
                pos = t.get('pos')
                break

        results.append({
            "word": word,
            "uploaded_count": u_count,
            "sample_count": c_count,
            "chi2": float(chi2),
            "p_value": float(p),
            "tfidf_score": float(uploaded_tfidf.get(word, 0)),
            "pos": pos
        })

    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    return {
        "results": results_sorted[:top_n],
        "uploaded_total": int(sum(uploaded_counts.values())),
        "corpus_total": int(sum(corpus_counts.values())),
    }



def keyness_spacy(uploaded_text, corpus_text, top_n=20, filter_func=filter_content_words):
    tokens_uploaded = filter_func(uploaded_text)
    tokens_corpus = filter_func(corpus_text)

    words_uploaded = [t['word'] if isinstance(t, dict) else t for t in tokens_uploaded]
    words_corpus = [t['word'] if isinstance(t, dict) else t for t in tokens_corpus]

    total_uploaded = len(words_uploaded)
    total_corpus = len(words_corpus)

    freq_uploaded = Counter(words_uploaded)
    freq_corpus = Counter(words_corpus)

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

        pos = None
        for t in tokens_uploaded:
            if isinstance(t, dict) and t['word'] == word:
                pos = t.get('pos')
                break

        results.append({
            "word": word,
            "uploaded_count": a,
            "sample_count": b,
            "chi2": float(chi2),
            "p_value": float(p),
            "log_likelihood": float(chi2),
            "effect_size": float(effect_size),
            "keyness": "Positive" if a > b else "Negative",
            "pos": pos
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

    # Extract words
    words_uploaded = [t['word'] if isinstance(t, dict) else t for t in tokens_uploaded]
    words_corpus = [t['word'] if isinstance(t, dict) else t for t in tokens_corpus]

    vectorizer = CountVectorizer(vocabulary=list(set(words_uploaded + words_corpus)))
    X = vectorizer.fit_transform([" ".join(words_uploaded), " ".join(words_corpus)])
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

        # Include POS if available
        pos = None
        for t in tokens_uploaded:
            if isinstance(t, dict) and t['word'] == word:
                pos = t.get('pos')
                break

        results.append({
            "word": word,
            "uploaded_count": int(u_count),
            "sample_count": int(c_count),
            "chi2": float(chi2),
            "p_value": float(p),
            "pos": pos
        })

    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    return {
        "results": results_sorted[:top_n],
        "uploaded_total": int(sum(uploaded_counts)),
        "corpus_total": int(sum(corpus_counts)),
    }
