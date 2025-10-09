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

# ------------------ NLTK Setup ------------------ #
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ------------------ Stopwords ------------------ #
def generate_roman_numerals(limit=1000):
    from roman import toRoman
    return {toRoman(i).lower() for i in range(1, limit + 1)}

ROMAN_STOPWORDS = generate_roman_numerals(1000)
CUSTOM_STOPWORDS = {"he", "she", "was", "for", "on", "as", "with", "at", "by", "an", "chapter"}
NUMBER_WORDS = {num2words(i) for i in range(1, 1001)}
NLTK_STOPWORDS = set(stopwords.words("english"))
ALL_STOPWORDS = NLTK_STOPWORDS.union(CUSTOM_STOPWORDS, NUMBER_WORDS, ROMAN_STOPWORDS)

# ------------------ Embedding Setup ------------------ #
EMBEDDING_BACKEND_CHOICES = ["conceptnet", "spacy"]
EMBEDDING_BACKEND = "conceptnet"

model = None
nlp = None
EMBEDDINGS_PATH = Path(__file__).resolve().parents[2] / "backend" / "data" / "numberbatch-en.txt"

def get_conceptnet_model():
    """Lazy-load ConceptNet embeddings."""
    global model
    if model is None:
        if not EMBEDDINGS_PATH.exists():
            from backend import download_embeddings
            EMBEDDINGS_PATH = download_embeddings.download_embeddings()
        print("Loading ConceptNet embeddings from:", EMBEDDINGS_PATH)
        model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH, binary=False)
        print(f"✅ ConceptNet embeddings loaded: {model.vector_size} dims")
    return model

# ------------------ spaCy Loader ------------------ #
def get_spacy_model():
    global nlp
    if nlp is None:
        import spacy
        try:
            nlp = spacy.load("en_core_web_md")
            print("✅ spaCy loaded successfully.")
        except OSError:
            print("❌ spaCy model not found. Run: python -m spacy download en_core_web_md")
            nlp = None
    return nlp

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
def get_vector(word):
    """Return vector for a word based on backend."""
    if EMBEDDING_BACKEND == "spacy" and nlp:
        lex = nlp.vocab[word]
        if lex.has_vector:
            return lex.vector
    elif EMBEDDING_BACKEND == "conceptnet":
        model_ref = get_conceptnet_model()
        if word in model_ref:
            return model_ref[word]
    return None

def suggest_theme(cluster_words, model_ref, backend="conceptnet"):
    if model_ref is None or not cluster_words:
        return "Unknown"
    theme_scores = {theme: 0.0 for theme in GENERAL_THEMES}
    for theme, keywords in GENERAL_THEMES.items():
        for kw in keywords:
            if backend == "conceptnet":
                if kw in model_ref:
                    for word in cluster_words:
                        if word in model_ref:
                            theme_scores[theme] += model_ref.similarity(word, kw)
            elif backend == "spacy":
                kw_token = model_ref(kw)[0]
                for word in cluster_words:
                    word_token = model_ref(word)[0]
                    if word_token.has_vector and kw_token.has_vector:
                        theme_scores[theme] += word_token.similarity(kw_token)
    return max(theme_scores, key=theme_scores.get)

# ------------------ Clustering Function ------------------ #
def cluster_text(text, top_words_per_cluster=10):
    global EMBEDDING_BACKEND
    vectors, valid_chunks, chunk_words = [], [], []
    chunks = [c.strip() for c in re.split(r'[.!?]\s+', text) if len(c.strip()) > 5]
    for chunk in chunks:
        tokens = nltk.word_tokenize(chunk)
        cleaned = [t.lower() for t in tokens if t.isalpha() and t.lower() not in ALL_STOPWORDS]
        if not cleaned:
            continue
        if EMBEDDING_BACKEND == "conceptnet":
            vecs = [get_vector(w) for w in cleaned if get_vector(w) is not None]
        elif EMBEDDING_BACKEND == "spacy" and nlp:
            doc = nlp(" ".join(cleaned))
            vecs = [token.vector for token in doc if token.has_vector and token.is_alpha]
            cleaned = [token.text.lower() for token in doc if token.has_vector and token.is_alpha]
        else:
            raise ValueError("Embeddings not loaded")
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
            valid_chunks.append(chunk)
            chunk_words.append(cleaned)
    if not vectors:
        return {"clusters": [], "top_terms": {}, "themes": {}, "num_clusters": 0, "num_docs": 0}
    vectors = np.array(vectors)
    n_docs = len(valid_chunks)
    num_clusters = min(20, max(2, n_docs // 50))
    reduced = PCA(n_components=min(50, vectors.shape[1])).fit_transform(vectors)
    labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(reduced)
    clusters = [
        {
            "label": int(lbl),
            "doc": doc,
            "words": words[:top_words_per_cluster],
            "x": float(r[0]),
            "y": float(r[1]),
        }
        for doc, words, lbl, r in zip(valid_chunks, chunk_words, labels, reduced)
    ]
    top_terms = {}
    for i in range(num_clusters):
        cluster_docs = [chunk_words[j] for j, lbl in enumerate(labels) if lbl == i]
        tokens = [t for doc in cluster_docs for t in doc]
        counts = Counter(tokens)
        top_terms[i] = [w for w, _ in counts.most_common(top_words_per_cluster)]
    embedding_ref = get_conceptnet_model() if EMBEDDING_BACKEND == "conceptnet" else get_spacy_model()
    themes = {i: suggest_theme(words, embedding_ref, backend=EMBEDDING_BACKEND) for i, words in top_terms.items()}
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
    global EMBEDDING_BACKEND
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=400)
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    text = data.get("text", "").strip()
    if not text:
        return JsonResponse({"error": "No text provided."}, status=400)
    backend_choice = data.get("embedding", "conceptnet").lower()
    if backend_choice not in EMBEDDING_BACKEND_CHOICES:
        return JsonResponse({"error": f"Invalid embedding choice: {backend_choice}"}, status=400)
    EMBEDDING_BACKEND = backend_choice
    # Load embeddings if needed
    try:
        if backend_choice == "conceptnet":
            get_conceptnet_model()
        elif backend_choice == "spacy":
            get_spacy_model()
            if nlp is None:
                return JsonResponse({"error": "spaCy model not loaded."}, status=500)
    except Exception as e:
        return JsonResponse({"error": f"Failed to load embeddings: {e}"}, status=500)
    # Perform clustering
    try:
        result = cluster_text(text, top_words_per_cluster=20)
        embedding_ref = get_conceptnet_model() if backend_choice == "conceptnet" else get_spacy_model()
        suggested = {cid: suggest_theme(words, embedding_ref, backend=backend_choice)
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