import re
import math
import gc
import os
import logging
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim import corpora, models
import numpy as np
from scipy.stats import chi2_contingency
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import spacy  # optional
except Exception:
    spacy = None
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from math import log

logger = logging.getLogger(__name__)

# OPTIMIZATION: Set chunk size for processing large texts
CHUNK_SIZE = 10000  # Process text in 10k character chunks


# Download required NLTK data on startup
def setup_nltk():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        logger.info("Downloading NLTK POS tagger...")
        nltk.download('averaged_perceptron_tagger', quiet=True)

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        logger.info("Downloading NLTK English POS tagger...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# Call setup on import
setup_nltk()


def log_memory_usage(label):
    """Log current memory usage."""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory at {label}: {memory_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not log memory: {e}")
    else:
        logger.debug(f"Checkpoint: {label} (psutil not available)")


def _safe_word_tokens(text):
    return re.findall(r"[A-Za-z]+(?:n't|'t|'re|'ve|'ll|'d|'m|'s)?", text)


def _safe_sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}


# ---------------------------
# Filtering functions (NLTK-ONLY for memory efficiency)
# ---------------------------

def filter_content_words(text):
    """
    Use NLTK only - much more memory efficient than spaCy.
    Returns content words only (NOUN, VERB, ADJ, ADV).
    """
    log_memory_usage("filter_content_words start")

    # Use NLTK for POS tagging
    try:
        tokens = [w for w in _safe_word_tokens(text) if len(w) > 2]

        # Process in chunks to avoid memory issues with very large texts
        chunk_size = 5000
        all_tagged = []

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            tagged_chunk = nltk.pos_tag(chunk_tokens)
            all_tagged.extend(tagged_chunk)

            # Clear memory after each chunk
            if i % (chunk_size * 4) == 0:
                gc.collect()

        POS_MAP_NLTK = {
            'NN': 'NOUN', 'NNS': 'NOUN',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
        }

        filtered = []
        for word, pos in all_tagged:
            if re.match(r"^[A-Za-z]+(n't|'t|'re|'ve|'ll|'d|'m|'s)?$", word):
                mapped = POS_MAP_NLTK.get(pos, "OTHER")
                if mapped in ALLOWED_POS:  # Only include content words
                    filtered.append({"word": word.lower(), "pos": mapped})

        del tokens, all_tagged
        gc.collect()

        log_memory_usage("filter_content_words end")
        logger.info(f"Filtered {len(filtered)} content words from text")
        return filtered

    except Exception as e:
        logger.error(f"Error in filter_content_words: {e}")
        # Fallback: simple tokenizer with no filtering
        tokens = _safe_word_tokens(text)
        filtered = []
        for word in tokens:
            if len(word) > 2 and re.match(r"^[a-zA-Z]+(n't|'t|'re|'ve|'ll|'d|'m|'s)?$", word):
                filtered.append({"word": word.lower(), "pos": "OTHER"})
        return filtered


def filter_all_words(text):
    """
    Use NLTK only - keep all alphabetic tokens > 2 chars with POS.
    """
    log_memory_usage("filter_all_words start")

    try:
        tokens = [w for w in _safe_word_tokens(text) if len(w) > 2]

        # Process in chunks
        chunk_size = 5000
        all_tagged = []

        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            tagged_chunk = nltk.pos_tag(chunk_tokens)
            all_tagged.extend(tagged_chunk)

            if i % (chunk_size * 4) == 0:
                gc.collect()

        POS_MAP_NLTK = {
            'NN': 'NOUN', 'NNS': 'NOUN',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
        }

        result = [{"word": w.lower(), "pos": POS_MAP_NLTK.get(p, "OTHER")} for w, p in all_tagged]

        del tokens, all_tagged
        gc.collect()

        log_memory_usage("filter_all_words end")
        logger.info(f"Filtered {len(result)} words from text")
        return result

    except Exception as e:
        logger.error(f"Error in filter_all_words: {e}")
        # Fallback
        tokens = _safe_word_tokens(text)
        return [{"word": w.lower(), "pos": "OTHER"} for w in tokens if len(w) > 2]


def extract_sentences(text, word):
    """
    Return all sentences from text containing the exact word (case-insensitive).
    Handles contractions properly - won't match partials.
    """
    if not text or not word:
        return []

    word_lower = word.lower()
    sentences = _safe_sentences(text)

    matched = []
    for s in sentences:
        tokens = _safe_word_tokens(s)
        if any(t.lower() == word_lower for t in tokens):
            matched.append(s)

    return matched

# ---------------------------
# Keyness Functions (with memory cleanup)
# ---------------------------

def keyness_nltk(uploaded_text, corpus_counts_map, top_n=50, filter_func=filter_content_words):
    """NLTK-style log-likelihood using counts-based corpus."""
    log_memory_usage("keyness_nltk start")

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
            "keyness_score": round(ll, 2),
            "direction": "Positive" if count_a > count_b else "Negative",
            "pos": pos
        })

    # OPTIMIZATION: Clear large objects
    del uploaded_tokens, uploaded_counts, all_words
    gc.collect()

    sorted_results = sorted(results, key=lambda x: x["keyness_score"], reverse=True)

    log_memory_usage("keyness_nltk end")
    return {"results": sorted_results[:top_n], "total_significant": len(sorted_results)}


def keyness_gensim(uploaded_text, corpus_counts_map, top_n=50, filter_func=filter_content_words):
    """Gensim-style TF-IDF + chi2 keyness."""
    log_memory_usage("keyness_gensim start")

    if filter_func is None:
        filter_func = lambda t: [{"word": w.lower(), "pos": "OTHER"} for t in t.split()]

    tokens_uploaded = filter_func(uploaded_text)
    words_uploaded = [t["word"] for t in tokens_uploaded]

    # Reconstruct "corpus text" from counts map (limit to prevent memory issues)
    words_corpus = []
    for word, count in corpus_counts_map.items():
        words_corpus.extend([word] * min(count, 1000))  # OPTIMIZATION: Cap repetitions

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
            "keyness_score": float(chi2_val),
            "direction": "Positive" if u_count > c_count else "Negative",
            "pos": pos
        })

    # OPTIMIZATION: Clear memory
    del tokens_uploaded, words_uploaded, words_corpus, dictionary, corpus_gensim, tfidf
    gc.collect()

    results_sorted = sorted(results, key=lambda x: x["keyness_score"], reverse=True)

    log_memory_usage("keyness_gensim end")
    return {
        "results": results_sorted[:top_n],
        "total_significant": len(results_sorted),
        "uploaded_total": uploaded_total,
        "corpus_total": corpus_total
    }


def keyness_spacy(uploaded_text, corpus_counts_map, top_n=50, filter_func=filter_content_words):
    """spaCy-based keyness analysis (uses NLTK filtering for memory efficiency)."""
    log_memory_usage("keyness_spacy start")

    if filter_func is None:
        filter_func = lambda t: [{"word": w.lower(), "pos": "OTHER"} for t in t.split()]

    tokens_uploaded = filter_func(uploaded_text)
    uploaded_counts = Counter([t["word"] for t in tokens_uploaded])
    corpus_counts = corpus_counts_map

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
            "log_likelihood": float(chi2),
            "effect_size": float(effect_size),
            "keyness_score": float(chi2),
            "direction": "Positive" if a > b else "Negative",
            "pos": pos
        })

    # OPTIMIZATION: Clear memory
    del tokens_uploaded, uploaded_counts
    gc.collect()

    sorted_results = sorted(results, key=lambda x: x["chi2"], reverse=True)

    log_memory_usage("keyness_spacy end")
    return {"results": sorted_results[:top_n], "total_significant": len(sorted_results)}


def keyness_sklearn(uploaded_text, corpus_counts_map, top_n=50, filter_func=None):
    """sklearn-based keyness analysis."""
    log_memory_usage("keyness_sklearn start")

    if filter_func is None:
        filter_func = lambda t: [{"word": w.lower(), "pos": "OTHER"} for t in t.split()]

    tokens_uploaded = filter_func(uploaded_text)
    uploaded_counts = Counter([t["word"] for t in tokens_uploaded])
    corpus_counts = corpus_counts_map

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
            "keyness_score": float(chi2),
            "direction": "Positive" if u_count > c_count else "Negative",
            "pos": pos
        })

    # OPTIMIZATION: Clear memory
    del tokens_uploaded, uploaded_counts
    gc.collect()

    sorted_results = sorted(results, key=lambda x: x["keyness_score"], reverse=True)

    log_memory_usage("keyness_sklearn end")
    return {"results": sorted_results[:top_n], "total_significant": len(sorted_results)}
