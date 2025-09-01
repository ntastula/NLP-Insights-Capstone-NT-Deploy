"""
sentiart_analyser.py

This file does the sentiment analysis.
It takes a text, splits it into words, looks up those words in our
CSV dictionary (sentiart_lexicon.csv), and then calculates:
- Overall average positivity/negativity (sentiment score)
- Average emotion strengths (joy, sadness, anger, fear, disgust)
- Which words are the strongest examples of each emotion
"""

# Import tools from Python
from pathlib import Path          # lets us handle file paths easily
import csv                        # lets us read CSV files
import re                         # lets us find words in text using patterns
from collections import Counter, defaultdict   # lets us count words and store sums

# Tell Python where the CSV lexicon file should live
LEXICON_PATH = Path(__file__).resolve().parent / "sentiart_lexicon.csv"

# These are the 5 emotions we are tracking
EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust"]

# This pattern will find "words" in text (just letters and apostrophes)
WORD_PATTERN = re.compile(r"[a-zA-Z']+")


# ----------------------------------------------------
# STEP 1: Break text into lowercase words
# ----------------------------------------------------
def split_text_into_words(text: str):
    """
    Take a string like "Happy Dogs!" and return ["happy", "dogs"].
    - .findall() gets all words that match WORD_PATTERN
    - .lower() makes sure words match our lexicon (which is lowercase)
    """
    if not text:
        return []
    words = WORD_PATTERN.findall(text)       # find words
    words_lower = [w.lower() for w in words] # make them lowercase
    return words_lower


# ----------------------------------------------------
# STEP 2: Load the lexicon from the CSV file
# ----------------------------------------------------
def load_lexicon():
    """
    Open the sentiart_lexicon.csv file and read all the rows into a dictionary.
    Each row has:
      word,sentiment_score,joy,sadness,anger,fear,disgust
    We store it like:
      {"happy": {"sentiment_score":0.8,"joy":0.9,...}, ...}
    """
    if not LEXICON_PATH.exists():
        # If the CSV file is missing, stop here.
        raise FileNotFoundError(f"Lexicon file not found at {LEXICON_PATH}")

    lexicon = {}  # empty dictionary to fill
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)   # read each line of CSV into a dictionary
        for row in reader:
            # Get the word from the row
            word = row.get("word", "").strip().lower()
            if not word:
                continue  # skip empty rows

            # Try to get sentiment score (positivity/negativity number)
            try:
                sentiment_score = float(row.get("sentiment_score", 0))
            except:
                sentiment_score = 0.0

            # Try to get emotion values
            emotions = {}
            for emotion in EMOTIONS:
                try:
                    emotions[emotion] = float(row.get(emotion, 0))
                except:
                    emotions[emotion] = 0.0

            # Save into lexicon dictionary
            lexicon[word] = {"sentiment_score": sentiment_score, **emotions}

    return lexicon


# ----------------------------------------------------
# STEP 3: Main analyser function
# ----------------------------------------------------
def analyse_with_sentiart(text: str):
    """
    This is the main function that views.py calls.
    It returns a dictionary that we can send back as JSON.
    """

    # --- Split text into words ---
    words = split_text_into_words(text)

    # --- Count how many times each word appears ---
    word_counts = Counter(words)   # example: {"happy": 2, "dog": 1}
    total_tokens = sum(word_counts.values())

    # --- Load the lexicon from CSV ---
    lexicon = load_lexicon()

    # --- Find which words in the text are also in the lexicon ---
    matched_words = []  # list of (word, count, scores)
    for word, count in word_counts.items():
        if word in lexicon:
            matched_words.append((word, count, lexicon[word]))

    # --- If no words matched, return empty results ---
    if not matched_words:
        return {
            "summary": {
                "sentiment_score_mean": 0.0,
                "token_count": total_tokens,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0
            },
            "emotions": {e: 0.0 for e in EMOTIONS},
            "top_contributors": {e: [] for e in EMOTIONS},
            "tokens": []
        }

    # --- Variables for totals ---
    tokens_output = []             # will hold per-word results
    sentiment_score_total = 0.0            # sum of sentiment score * count
    sentiment_score_count = 0              # total count for sentiment score
    emotion_totals = defaultdict(float)  # sum of emotion values * count
    emotion_count = 0              # total count for emotions
    positive_total = 0             # number of positive words
    negative_total = 0             # number of negative words

    # --- Go through each matched word ---
    for word, count, scores in matched_words:
        # sentiment score score for this word
        sentiment_score = scores["sentiment_score"]

        # emotion scores for this word
        emotion_scores = {e: scores[e] for e in EMOTIONS}

        # Add to the list of tokens we will return
        tokens_output.append({
            "word": word,
            "count": count,
            "sentiment_score": sentiment_score,
            "emotions": emotion_scores
        })

        # Update totals for overall averages
        sentiment_score_total += count * sentiment_score
        sentiment_score_count += count
        for e in EMOTIONS:
            emotion_totals[e] += count * emotion_scores[e]
        emotion_count += count

        # Count positive and negative words separately
        if sentiment_score > 0:
            positive_total += count
        elif sentiment_score < 0:
            negative_total += count

    # --- Work out averages ---
    sentiment_score_mean = sentiment_score_total / sentiment_score_count if sentiment_score_count else 0.0
    emotion_avgs = {e: emotion_totals[e] / emotion_count if emotion_count else 0.0 for e in EMOTIONS}
    positive_ratio = positive_total / sentiment_score_count if sentiment_score_count else 0.0
    negative_ratio = negative_total / sentiment_score_count if sentiment_score_count else 0.0

    # --- Find top words for each emotion ---
    top_words = {}
    for e in EMOTIONS:
        # Sort tokens by emotion score and frequency
        scored = sorted(
            [(t["word"], t["emotions"][e], t["count"]) for t in tokens_output],
            key=lambda x: (x[1], x[2]),
            reverse=True
        )
        # Only keep the first 10
        unique = []
        seen = set()
        for word, score, count in scored:
            if word in seen:
                continue
            seen.add(word)
            unique.append({"word": word, "score": score, "count": count})
            if len(unique) >= 10:
                break
        top_words[e] = unique

    # --- Final dictionary to return ---
    return {
        "summary": {
            "sentiment_score_mean": sentiment_score_mean,
            "token_count": total_tokens,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio
        },
        "emotions": emotion_avgs,
        "top_contributors": top_words,
        "tokens": tokens_output
    }
