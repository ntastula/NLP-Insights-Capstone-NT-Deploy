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
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer

# Avoid import-time downloads and heavy loads
DOWNLOAD_EXTERNAL_MODELS = False

# ------------------ Tokenization & Stopwords ------------------ #
# Lightweight tokenization: words and common contractions
SAFE_WORD_RE = r"[A-Za-z]+(?:n't|'t|'re|'ve|'ll|'d|'m|'s)?"


def safe_word_tokens(text):
    return re.findall(SAFE_WORD_RE, text)


# Stopwords fallback (small static set)
STATIC_STOPWORDS = {
    'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'on', 'for', 'as', 'with', 'at', 'by', 'an', 'be', 'this',
    'from', 'or', 'are', 'was', 'were', 'but', 'not', 'have', 'has', 'had', 'which', 'you', 'your', 'their', 'his',
    'her', 'its', 'they', 'them', 'we', 'our', 'us', 'i', 'me', 'my'
}

try:
    from nltk.corpus import stopwords as nltk_stopwords

    NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
except Exception:
    NLTK_STOPWORDS = STATIC_STOPWORDS


def generate_roman_numerals(limit=1000):
    from roman import toRoman
    return {toRoman(i).lower() for i in range(1, limit + 1)}


ROMAN_STOPWORDS = generate_roman_numerals(1000)
CUSTOM_STOPWORDS = {"he", "she", "was", "for", "on", "as", "with", "at", "by", "an", "chapter"}
NUMBER_WORDS = {num2words(i) for i in range(1, 1001)}
ALL_STOPWORDS = NLTK_STOPWORDS.union(CUSTOM_STOPWORDS, NUMBER_WORDS, ROMAN_STOPWORDS)

# ------------------ Embeddings Setup ------------------ #
model = None  # ConceptNet model

# Paths for ConceptNet file if pre-provisioned
DATA_DIR = Path(__file__).resolve().parents[2] / "backend" / "data"
EMBEDDINGS_PATH = DATA_DIR / "numberbatch-en.txt"

# ------------------ General Themes ------------------ #
GENERAL_THEMES = {
    "Nature": ["forest", "river", "mountain", "snow", "sky", "earth", "wind", "tree", "sun",
               "moon", "star", "ocean", "sea", "rain", "storm", "cloud", "flower", "garden"],

    "Movement": ["walk", "run", "jump", "chase", "move", "stumble", "dance", "ride", "wander",
                 "travel", "journey", "climb", "fall", "fly", "swim", "escape", "approach"],

    "Emotions": ["fear", "joy", "love", "anger", "hope", "sorrow", "desire", "happy", "sad",
                 "anxious", "lonely", "excited", "disappointed", "proud", "ashamed", "confused",
                 "surprised", "grateful", "envious", "disgusted"],

    "Characters": ["man", "woman", "child", "friend", "stranger", "family", "hero", "villain",
                   "mother", "father", "brother", "sister", "leader", "follower", "enemy", "ally"],

    "Time": ["night", "morning", "dawn", "evening", "hour", "day", "moment", "week", "month",
             "year", "past", "present", "future", "yesterday", "tomorrow", "always", "never"],

    "Communication": ["say", "ask", "shout", "whisper", "call", "answer", "speak", "tell",
                      "listen", "hear", "reply", "argue", "discuss", "explain", "persuade"],

    "Conflict": ["fight", "battle", "struggle", "resist", "oppose", "defend", "attack", "war",
                 "argue", "conflict", "tension", "challenge", "confront", "compete"],

    "Cognition": ["think", "know", "believe", "remember", "forget", "understand", "wonder",
                  "imagine", "realize", "consider", "decide", "doubt", "dream", "recognize"],

    "Sensation": ["see", "look", "watch", "hear", "listen", "touch", "feel", "smell", "taste",
                  "sense", "perceive", "observe", "notice", "aware"],

    "Space/Location": ["house", "home", "room", "city", "village", "place", "inside", "outside",
                       "above", "below", "near", "far", "here", "there", "where", "path", "road"],

    "Social Relationships": ["marry", "divorce", "betray", "trust", "befriend", "unite", "separate",
                             "meet", "leave", "join", "abandon", "support", "help", "harm"],

    "Power/Authority": ["king", "queen", "lord", "master", "servant", "ruler", "power", "control",
                        "command", "obey", "rule", "govern", "lead", "follow", "submit"],

    "Morality/Values": ["good", "evil", "right", "wrong", "just", "unjust", "moral", "honest",
                        "lie", "truth", "virtue", "sin", "honor", "shame", "duty", "guilt"],

    "Death/Life": ["life", "death", "die", "live", "born", "birth", "survive", "kill", "dead",
                   "alive", "mortal", "immortal", "grave", "funeral", "resurrection"],

    "Change/Transformation": ["change", "transform", "become", "grow", "evolve", "develop", "shift",
                              "adapt", "convert", "alter", "turn", "emerge", "transition"],

    "Abstract Concepts": ["freedom", "justice", "beauty", "truth", "wisdom", "knowledge", "fate",
                          "destiny", "luck", "chance", "purpose", "meaning", "soul", "spirit"],

    "Technology/Modernity": ["computer", "phone", "internet", "machine", "device", "digital",
                             "technology", "modern", "electric", "automatic", "online"],

    "Economy/Commerce": ["money", "buy", "sell", "trade", "business", "work", "job", "pay",
                         "rich", "poor", "wealth", "price", "cost", "value", "invest"],

    "Religion/Spirituality": ["god", "pray", "worship", "faith", "believe", "church", "temple",
                              "sacred", "holy", "divine", "ritual", "blessing", "curse", "sin"],

    "Education/Learning": ["learn", "teach", "study", "school", "student", "teacher", "lesson",
                           "educate", "train", "practice", "master", "knowledge", "wisdom"],

    "Health/Body": ["body", "hand", "eye", "heart", "blood", "pain", "sick", "heal", "wound",
                    "healthy", "strong", "weak", "tired", "energy", "medicine", "doctor"],

    "Food/Sustenance": ["eat", "drink", "food", "meal", "hungry", "thirst", "cook", "feast",
                        "bread", "water", "wine", "fruit", "meat", "taste", "devour"],
}


# ------------------ Helper Functions ------------------ #
def suggest_theme(cluster_words, model):
    """Suggest a theme for a cluster based on ConceptNet similarity."""
    if not cluster_words:
        return "Unknown"

    theme_scores = {theme: 0.0 for theme in GENERAL_THEMES}

    if model is not None:
        # Use ConceptNet similarity
        for theme, keywords in GENERAL_THEMES.items():
            for kw in keywords:
                if kw in model:
                    for word in cluster_words:
                        if word in model:
                            theme_scores[theme] += model.similarity(word, kw)
    else:
        # Fallback: overlap score
        for theme, keywords in GENERAL_THEMES.items():
            overlap = len(set(cluster_words) & set(keywords))
            theme_scores[theme] += overlap

    return max(theme_scores, key=theme_scores.get)


# ------------------ Clustering ------------------ #
def cluster_text(text, top_words_per_cluster=10):
    """
    Cluster text into groups using ConceptNet embeddings if available, otherwise TF-IDF fallback.
    Each cluster point includes a 'words' array for display in scatterplots.
    """
    global model

    if not text.strip():
        return {"clusters": [], "top_terms": {}, "themes": {}, "num_clusters": 0, "num_docs": 0}

    # ------------------ Split into chunks and clean ------------------ #
    vectors, valid_chunks, chunk_words = [], [], []
    chunks = [c.strip() for c in re.split(r'[.!?]\s+', text) if len(c.strip()) > 5]

    for chunk in chunks:
        tokens = safe_word_tokens(chunk)
        cleaned = [t.lower() for t in tokens if re.match(r"^[A-Za-z]+$", t) and t.lower() not in ALL_STOPWORDS]
        if not cleaned:
            continue
        valid_chunks.append(chunk)
        chunk_words.append(cleaned)

    if not chunk_words:
        return {"clusters": [], "top_terms": {}, "themes": {}, "num_clusters": 0, "num_docs": 0}

    # ------------------ Build vectors ------------------ #
    use_vectors = False
    if model is not None:
        for cleaned in chunk_words:
            vecs = [model[w] for w in cleaned if w in model]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
        use_vectors = len(vectors) == len(chunk_words)

    # TF-IDF fallback if ConceptNet is unavailable
    if not use_vectors:
        docs = [" ".join(words) for words in chunk_words]
        tfidf = TfidfVectorizer(token_pattern=SAFE_WORD_RE, min_df=1)
        X = tfidf.fit_transform(docs)
        vectors = X.toarray()

    n_docs = len(valid_chunks)

    # ------------------ Determine number of clusters ------------------ #
    if n_docs < 20:
        num_clusters = 2
    elif n_docs < 100:
        num_clusters = 3
    elif n_docs < 300:
        num_clusters = 5
    elif n_docs < 1000:
        num_clusters = 10
    else:
        num_clusters = min(20, n_docs // 200)
    # Clamp to valid range to avoid KMeans errors
    num_clusters = max(1, min(num_clusters, n_docs))

    # ------------------ PCA for dimensionality reduction ------------------ #
    # Use at most 10 components and respect sample count; then keep first 2 for plotting
    vectors = np.asarray(vectors)
    n_components = min(10, vectors.shape[1], vectors.shape[0])
    if n_components < 2:
        if vectors.shape[1] > 0:
            reduced = np.column_stack([vectors[:, 0], np.zeros((vectors.shape[0],))])
        else:
            reduced = np.zeros((vectors.shape[0], 2))
    else:
        reduced = PCA(n_components=n_components, random_state=42).fit_transform(vectors)
        if reduced.shape[1] >= 2:
            reduced = reduced[:, :2]

    # ------------------ KMeans clustering ------------------ #
    labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(reduced)
    clusters = [
        {
            "label": int(lbl),
            "doc": doc,
            "words": words[:top_words_per_cluster],  # keep only top N words per chunk
            "x": float(r[0]),  # include PCA coords for plotting
            "y": float(r[1]),
        }
        for doc, words, lbl, r in zip(valid_chunks, chunk_words, labels, reduced)
    ]

    # ------------------ Top terms per cluster ------------------ #
    top_terms = {}
    for i in range(num_clusters):
        cluster_docs = [chunk_words[j] for j, lbl in enumerate(labels) if lbl == i]
        tokens = [t for doc in cluster_docs for t in doc]
        counts = Counter(tokens)
        top_terms[i] = [w for w, _ in counts.most_common(top_words_per_cluster)]

    # ------------------ Suggest themes ------------------ #
    themes = {i: suggest_theme(words, model) for i, words in top_terms.items()}

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

    # Load ConceptNet embeddings if not already loaded
    if model is None and EMBEDDINGS_PATH.exists():
        try:
            print("Loading ConceptNet embeddings...")
            model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=False)
            print("✅ ConceptNet embeddings loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load ConceptNet embeddings: {e}")
            print("⚠️ Falling back to TF-IDF embeddings.")
            model = None

    # Perform clustering (uses TF-IDF fallback if ConceptNet unavailable)
    try:
        result = cluster_text(text, top_words_per_cluster=20)

        # Suggest themes
        suggested = {cid: suggest_theme(words, model)
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
