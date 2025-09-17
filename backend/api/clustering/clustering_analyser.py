import os
import re
import json
import traceback
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings

import nltk
from nltk.corpus import stopwords
from num2words import num2words

# ------------------ Setup NLTK ------------------ #
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

for corpus in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus, download_dir=NLTK_DATA_DIR, quiet=True)

CUSTOM_STOPWORDS = {"he","she","was","for","on","as","with","at","by","an"}
NUMBER_WORDS = {num2words(i) for i in range(1, 1001)} # Generate spelled-out numbers up to 1000
NLTK_STOPWORDS = set(stopwords.words("english"))
ALL_STOPWORDS = NLTK_STOPWORDS.union(CUSTOM_STOPWORDS).union(NUMBER_WORDS)

# ------------------ Load Embeddings Locally Only ------------------ #
EMBEDDINGS_PATH = os.path.join(settings.BASE_DIR, "models", "numberbatch-en.txt")

model = None
if os.path.exists(EMBEDDINGS_PATH):
    try:
        print("Loading embeddings from:", EMBEDDINGS_PATH)
        model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=False)
        print("Embeddings loaded! Dimensions:", model.vector_size)
    except Exception as e:
        print("Failed to load embeddings:", e)
else:
    print(
        f"Embeddings file not found at {EMBEDDINGS_PATH}.\n"
        "This file is large and should NOT be committed to GitHub.\n"
        "Download it separately if you want full clustering features:\n"
        "https://github.com/commonsense/conceptnet-numberbatch"
    )

# ------------------ General Themes ------------------ #
GENERAL_THEMES = {
    "Nature": ["forest", "river", "mountain", "snow", "sky", "earth", "wind", "tree", "sun"],
    "Movement": ["walk", "run", "jump", "chase", "move", "stumble", "dance", "ride", "wander"],
    "Emotions": ["fear", "joy", "love", "anger", "hope", "sorrow", "desire", "happy", "sad"],
    "Characters": ["man", "woman", "child", "friend", "stranger", "family", "hero", "villain"],
    "Time": ["night", "morning", "dawn", "evening", "hour", "day", "moment"],
    "Objects": ["book", "letter", "sword", "key", "lamp", "chair", "door", "tool"],
    "Communication": ["say", "ask", "shout", "whisper", "call", "answer", "speak"],
}

def suggest_theme(cluster_words, model, top_n=3):
    if model is None or not cluster_words:
        return "Unknown"
    theme_scores = {theme: 0.0 for theme in GENERAL_THEMES}
    for theme, keywords in GENERAL_THEMES.items():
        for kw in keywords:
            if kw in model:
                for word in cluster_words:
                    if word in model:
                        theme_scores[theme] += model.similarity(word, kw)
    return max(theme_scores, key=theme_scores.get)

# ------------------ Clustering ------------------ #
def cluster_text(text, model=None, top_words_per_cluster=10):
    chunks = [c.strip() for c in re.split(r'[.!?]\s+', text) if len(c.strip()) > 5]

    vectors, valid_chunks = [], []
    for chunk in chunks:
        tokens = nltk.word_tokenize(chunk)
        cleaned = [t.lower() for t in tokens if t.isalpha() and t.lower() not in ALL_STOPWORDS]
        if not cleaned:
            continue
        vecs = [model[w] for w in cleaned if model and w in model]
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
            valid_chunks.append(chunk)

    if not vectors:
        return {"clusters": [], "top_terms": {}, "themes": {}, "num_clusters": 0, "num_docs": 0}

    vectors = np.array(vectors)
    n_docs = len(valid_chunks)
    if n_docs < 20: num_clusters = 2
    elif n_docs < 100: num_clusters = 3
    elif n_docs < 300: num_clusters = 5
    elif n_docs < 1000: num_clusters = 10
    else: num_clusters = min(20, n_docs // 200)

    reduced = PCA(n_components=min(50, vectors.shape[1])).fit_transform(vectors)
    labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(reduced)

    clusters = [{"label": int(lbl), "doc": doc} for doc, lbl in zip(valid_chunks, labels)]

    top_terms = {}
    for i in range(num_clusters):
        cluster_docs = [valid_chunks[j] for j, lbl in enumerate(labels) if lbl == i]
        tokens = []
        for doc in cluster_docs:
            tokens.extend([t.lower() for t in nltk.word_tokenize(doc) if t.isalpha() and t.lower() not in ALL_STOPWORDS])
        counts = Counter(tokens)
        top_terms[i] = [w for w, _ in counts.most_common(top_words_per_cluster)]

    themes = {}
    for i, words in top_terms.items():
        if any(w in words for w in ["see","look","saw","noticed","viewed","eyes"]):
            themes[i] = "Observation / Vision"
        elif any(w in words for w in ["walk","run","go","came","moved","step"]):
            themes[i] = "Movement / Action"
        elif any(w in words for w in ["nature","tree","river","sky","earth"]):
            themes[i] = "Nature / Setting"
        elif any(w in words for w in ["said","asked","told","voice","speak"]):
            themes[i] = "Dialogue / Communication"
        else:
            themes[i] = "General Theme"

    return {
        "clusters": clusters,
        "top_terms": top_terms,
        "themes": themes,
        "num_clusters": num_clusters,
        "num_docs": n_docs,
    }

# ------------------ Django Endpoint ------------------ #
@csrf_exempt
def clustering_analysis(request):
    print("Request method:", request.method)
    print("Request body:", request.body)
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=400)

    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    text = data.get("text", "").strip()
    if not text:
        return JsonResponse({"error": "No text provided."}, status=400)

    if model is None:
        return JsonResponse({"error": "Embeddings not loaded. Place 'numberbatch-en.txt' in backend/models to enable clustering."}, status=500)

    try:
        result = cluster_text(text, model=model, top_words_per_cluster=20)
        suggested = {cid: suggest_theme(words, model) for cid, words in result["top_terms"].items()}

        return JsonResponse({
            "clusters": result["clusters"],
            "top_terms": result["top_terms"],
            "suggested_themes": suggested,
            "num_clusters": result["num_clusters"],
            "num_docs": result["num_docs"],
        })
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

