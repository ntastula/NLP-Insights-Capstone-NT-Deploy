# api/keyness/keyness_analyser.py
import re
import math
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

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




# import math
# import nltk
# from collections import Counter
# from nltk.tokenize import word_tokenize
#
# # Download tokenizer model (quiet avoids startup spam)
# nltk.download("punkt", quiet=True)
#
#
# def compute_keyness(text_a: str, text_b: str, top_n: int = 20):
#     """Compute log-likelihood keyness between two texts."""
#     tokens_a = tokenize(text_a)
#     tokens_b = tokenize(text_b)
#
#     freq_a = Counter(tokens_a)
#     freq_b = Counter(tokens_b)
#
#     results = []
#     for word in set(freq_a.keys()).union(freq_b.keys()):
#         a = freq_a.get(word, 0)
#         b = freq_b.get(word, 0)
#         if a + b < 2:
#             continue
#
#         c = len(tokens_a) - a
#         d = len(tokens_b) - b
#
#         try:
#             e1 = c * (a + b) / (c + d)
#             e2 = d * (a + b) / (c + d)
#             ll = 2 * ((a * math.log(a / e1)) + (b * math.log(b / e2)))
#         except (ValueError, ZeroDivisionError):
#             ll = 0.0
#
#         results.append({
#             "word": str(word),
#             "log_likelihood": float(ll),
#             "count_a": int(a),
#             "count_b": int(b)
#         })
#
#     return sorted(results, key=lambda x: x["log_likelihood"], reverse=True)[:int(top_n)]
#
#
# def tokenize(text: str):
#     """Lowercase + tokenize a text string."""
#     return [w.lower() for w in word_tokenize(text)]










# import re
# import math
# from collections import Counter
# import numpy as np
#
# # Libraries
# import nltk
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#
# # Optional imports (don’t break if not used)
# try:
#     import spacy
#     nlp = spacy.load("en_core_web_sm")
# except Exception:
#     nlp = None
#
# try:
#     from gensim import corpora, models
# except Exception:
#     corpora, models = None, None
#
# try:
#     from corpus_toolkit import corpus_tools as ct
# except ImportError:
#     ct = None  # Optional
#
# # Ensure NLTK tokenizer is available
# nltk.download("punkt", quiet=True)
#
#
# class KeynessAnalyser:
#     def __init__(self, text_a, text_b):
#         self.text_a = text_a
#         self.text_b = text_b
#
#     # ---------- Helper ----------
#     def _clean(self, text):
#         return re.sub(r"[^\w\s]", " ", text.lower())
#
#     # ---------- NLTK ----------
#     def analyse_nltk(self, top_n=20):
#         tokens_a = word_tokenize(self._clean(self.text_a))
#         tokens_b = word_tokenize(self._clean(self.text_b))
#
#         freq_a = Counter(tokens_a)
#         freq_b = Counter(tokens_b)
#
#         results = []
#         for word in set(freq_a.keys()).union(freq_b.keys()):
#             a, b = freq_a.get(word, 0), freq_b.get(word, 0)
#             if a + b < 2:
#                 continue
#             c, d = len(tokens_a) - a, len(tokens_b) - b
#             try:
#                 e1 = c * (a + b) / (c + d)
#                 e2 = d * (a + b) / (c + d)
#                 ll = 2 * ((a * math.log(a / e1)) + (b * math.log(b / e2)))
#             except (ValueError, ZeroDivisionError):
#                 ll = 0
#             results.append({
#                 "word": word,
#                 "log_likelihood": float(ll),
#                 "count_a": int(a),
#                 "count_b": int(b)
#             })
#         return sorted(results, key=lambda x: x["log_likelihood"], reverse=True)[:top_n]
#
#     # ---------- Corpus Toolkit ----------
#     def analyse_corpus_toolkit(self, top_n=20):
#         if ct is None:
#             return [{"error": "corpus-toolkit not installed"}]
#         freq_a = ct.frequency(self.text_a)
#         freq_b = ct.frequency(self.text_b)
#         keyness = ct.keyness(freq_a, freq_b)
#         # Convert to list of dicts
#         results = [{"word": w, "score": float(s)} for w, s in keyness[:top_n]]
#         return results
#
#     # ---------- Scikit-learn ----------
#     def analyse_sklearn(self, top_n=20):
#         vectorizer = CountVectorizer(stop_words="english")
#         X = vectorizer.fit_transform([self.text_a, self.text_b])
#         words = vectorizer.get_feature_names_out()
#         counts = X.toarray()
#
#         diff = counts[0] - counts[1]
#         ranked = sorted(zip(words, diff), key=lambda x: abs(x[1]), reverse=True)
#         return [{"word": w, "diff": int(d)} for w, d in ranked[:top_n]]
#
#     def analyse_sklearn_tfidf(self, top_n=20):
#         vectorizer = TfidfVectorizer(stop_words="english")
#         X = vectorizer.fit_transform([self.text_a, self.text_b])
#         words = vectorizer.get_feature_names_out()
#         tfidf_scores = X.toarray()
#         diff = tfidf_scores[0] - tfidf_scores[1]
#         ranked = sorted(zip(words, diff), key=lambda x: abs(x[1]), reverse=True)
#         return [{"word": w, "diff": float(d)} for w, d in ranked[:top_n]]
#
#     # ---------- Gensim ----------
#     def analyse_gensim(self, top_n=20):
#         texts = [self._clean(self.text_a).split(), self._clean(self.text_b).split()]
#         dictionary = corpora.Dictionary(texts)
#         corpus = [dictionary.doc2bow(text) for text in texts]
#
#         lda = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)
#         topics = lda.print_topics(num_words=10)
#         # Convert topics to list of dicts
#         results = []
#         for i, t in topics:
#             results.append({"topic_id": int(i), "words": t})
#         return results[:top_n]
#
#     # ---------- spaCy ----------
#     def analyse_spacy(self, top_n=20):
#         doc_a, doc_b = nlp(self.text_a), nlp(self.text_b)
#         lemmas_a = [token.lemma_.lower() for token in doc_a if token.is_alpha]
#         lemmas_b = [token.lemma_.lower() for token in doc_b if token.is_alpha]
#         freq_a, freq_b = Counter(lemmas_a), Counter(lemmas_b)
#
#         diff = {w: freq_a[w] - freq_b.get(w, 0) for w in freq_a}
#         ranked = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)
#         return [{"word": w, "diff": int(d)} for w, d in ranked[:top_n]]
