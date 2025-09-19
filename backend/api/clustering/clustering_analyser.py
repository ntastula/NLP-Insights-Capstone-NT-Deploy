import os
import re
import json
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from pathlib import Path
from gensim.models import KeyedVectors

import nltk
import spacy
from nltk.corpus import stopwords
from num2words import num2words
from backend import download_embeddings

# ------------------ NLTK Setup ------------------ #
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

CUSTOM_STOPWORDS = {"he","she","was","for","on","as","with","at","by","an", "chapter"}
NUMBER_WORDS = {num2words(i) for i in range(1, 1001)}
NLTK_STOPWORDS = set(stopwords.words("english"))
ALL_STOPWORDS = NLTK_STOPWORDS.union(CUSTOM_STOPWORDS).union(NUMBER_WORDS)

# ------------------ Embeddings Setup ------------------ #
EMBEDDING_BACKEND_CHOICES = ["conceptnet", "spacy"]
EMBEDDING_BACKEND = "conceptnet"

model = None
nlp = None
if EMBEDDING_BACKEND == "spacy":
    import spacy
    try:
        nlp = spacy.load("en_core_web_md")
        print("✅ spaCy loaded successfully.")
    except OSError:
        print("❌ spaCy model not found. Run: python -m spacy download en_core_web_md")
        nlp = None
else:
    # ConceptNet path
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parents[2] / "backend" / "data"
    EMBEDDINGS_PATH = DATA_DIR / "numberbatch-en.txt"
    print("Looking for embeddings at:", EMBEDDINGS_PATH)
    print("Exists?", EMBEDDINGS_PATH.exists())
    if EMBEDDINGS_PATH.exists():
        model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=False)
        print(f"✅ ConceptNet embeddings loaded: {model.vector_size} dims")
    else:
        print("❌ ConceptNet embeddings not found. Download using download_embeddings.py")

# ------------------ General Themes ------------------ #
GENERAL_THEMES = {
    "Nature": ["forest","river","mountain","snow","sky","earth","wind","tree","sun"],
    "Movement": ["walk","run","jump","chase","move","stumble","dance","ride","wander"],
    "Emotions": ["fear","joy","love","anger","hope","sorrow","desire","happy","sad"],
    "Characters": ["man","woman","child","friend","stranger","family","hero","villain"],
    "Time": ["night","morning","dawn","evening","hour","day","moment"],
    "Objects": ["book","letter","sword","key","lamp","chair","door","tool"],
    "Communication": ["say","ask","shout","whisper","call","answer","speak"],
}

# ------------------ Helper Functions ------------------ #
def get_vector(word):
    """Return vector for a word based on backend."""
    if EMBEDDING_BACKEND == "spacy" and nlp:
        lex = nlp.vocab[word]
        if lex.has_vector:
            return lex.vector
    elif EMBEDDING_BACKEND == "conceptnet" and model and word in model:
        return model[word]
    return None


def suggest_theme(cluster_words, model, backend="conceptnet"):
    if model is None or not cluster_words:
        return "Unknown"

    theme_scores = {theme: 0.0 for theme in GENERAL_THEMES}

    for theme, keywords in GENERAL_THEMES.items():
        for kw in keywords:
            if backend == "conceptnet":
                if kw in model:
                    for word in cluster_words:
                        if word in model:
                            theme_scores[theme] += model.similarity(word, kw)
            elif backend == "spacy":
                # Use spaCy similarity
                kw_token = model(kw)[0]  # convert keyword to token
                for word in cluster_words:
                    word_token = model(word)[0]
                    if word_token.has_vector and kw_token.has_vector:
                        theme_scores[theme] += word_token.similarity(kw_token)

    return max(theme_scores, key=theme_scores.get)

# ------------------ Clustering ------------------ #
def cluster_text(text, top_words_per_cluster=10):
    """
    Cluster text into groups using either ConceptNet or spaCy embeddings.
    Chooses backend automatically based on EMBEDDING_BACKEND global variable.
    """
    global EMBEDDING_BACKEND, model, nlp

    if not text.strip():
        return {"clusters": [], "top_terms": {}, "themes": {}, "num_clusters": 0, "num_docs": 0}

    # ------------------ Tokenize and get vectors ------------------ #
    vectors, valid_chunks = [], []
    chunks = [c.strip() for c in re.split(r'[.!?]\s+', text) if len(c.strip()) > 5]

    for chunk in chunks:
        tokens = nltk.word_tokenize(chunk)
        cleaned = [t.lower() for t in tokens if t.isalpha() and t.lower() not in ALL_STOPWORDS]
        if not cleaned:
            continue

        # Get vectors depending on backend
        if EMBEDDING_BACKEND == "conceptnet" and model:
            vecs = [model[w] for w in cleaned if w in model]
        elif EMBEDDING_BACKEND == "spacy" and nlp:
            doc = nlp(" ".join(cleaned))
            vecs = [token.vector for token in doc if token.has_vector and token.is_alpha]
            cleaned = [token.text.lower() for token in doc if token.has_vector and token.is_alpha]
        else:
            raise ValueError("Embeddings not loaded")

        if vecs:
            vectors.append(np.mean(vecs, axis=0))
            valid_chunks.append(chunk)

    if not vectors:
        return {"clusters": [], "top_terms": {}, "themes": {}, "num_clusters": 0, "num_docs": 0}

    vectors = np.array(vectors)
    n_docs = len(valid_chunks)

    # ------------------ Determine number of clusters ------------------ #
    if n_docs < 20: num_clusters = 2
    elif n_docs < 100: num_clusters = 3
    elif n_docs < 300: num_clusters = 5
    elif n_docs < 1000: num_clusters = 10
    else: num_clusters = min(20, n_docs // 200)

    # ------------------ PCA for dimensionality reduction ------------------ #
    reduced = PCA(n_components=min(50, vectors.shape[1])).fit_transform(vectors)

    # ------------------ KMeans clustering ------------------ #
    labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(reduced)
    clusters = [{"label": int(lbl), "doc": doc} for doc, lbl in zip(valid_chunks, labels)]

    # ------------------ Top terms per cluster ------------------ #
    top_terms = {}
    for i in range(num_clusters):
        cluster_docs = [valid_chunks[j] for j, lbl in enumerate(labels) if lbl == i]
        tokens = []
        for doc in cluster_docs:
            tokens.extend([t.lower() for t in nltk.word_tokenize(doc) if t.isalpha() and t.lower() not in ALL_STOPWORDS])
        counts = Counter(tokens)
        top_terms[i] = [w for w, _ in counts.most_common(top_words_per_cluster)]

    # ------------------ Suggest themes ------------------ #
    embedding_ref = model or nlp
    themes = {i: suggest_theme(words, embedding_ref) for i, words in top_terms.items()}

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
    global model
    global nlp
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=400)

    # Parse JSON request
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    text = data.get("text", "").strip()
    if not text:
        return JsonResponse({"error": "No text provided."}, status=400)

    # Choose embedding backend
    backend_choice = data.get("embedding", "conceptnet").lower()
    if backend_choice not in EMBEDDING_BACKEND_CHOICES:
        return JsonResponse({"error": f"Invalid embedding choice: {backend_choice}"}, status=400)

    global EMBEDDING_BACKEND
    EMBEDDING_BACKEND = backend_choice

    # Load embeddings accordingly
    try:
        if backend_choice == "conceptnet":
            embeddings_path = download_embeddings.download_embeddings()
            if not model:
                # Load ConceptNet model only if not already loaded
                # global model
                model = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

        elif backend_choice == "spacy":
            if not nlp:
                import spacy
                try:
                    # global nlp
                    nlp = spacy.load("en_core_web_md")
                except OSError:
                    return JsonResponse({
                        "error": "spaCy model not found. Run: python -m spacy download en_core_web_md"
                    }, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load embeddings: {e}"}, status=500)

    # Ensure we have embeddings loaded
    if (backend_choice == "conceptnet" and not model) or (backend_choice == "spacy" and not nlp):
        return JsonResponse({"error": "Embeddings not loaded."}, status=500)

    # Perform clustering
    try:
        result = cluster_text(text, top_words_per_cluster=20)

        # Determine which embedding backend to use for suggesting themes
        if backend_choice == "conceptnet":
            suggested = {cid: suggest_theme(words, model, backend="conceptnet")
                         for cid, words in result["top_terms"].items()}
        elif backend_choice == "spacy":
            suggested = {cid: suggest_theme(words, nlp, backend="spacy")
                         for cid, words in result["top_terms"].items()}

        return JsonResponse({
            "clusters": result["clusters"],
            "top_terms": result["top_terms"],
            "suggested_themes": suggested,
            "num_clusters": result["num_clusters"],
            "num_docs": result["num_docs"],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


