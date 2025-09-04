"""
sentiart_analyser.py

This file does the sentiment analysis.

It takes a text, splits it into tokens (words + simple emoticons), looks up those
tokens in our CSV dictionary (sentiart_lexicon.csv), and then calculates:

- Overall average positivity/negativity ("sentiment_score_mean")
- Separate "polarity" (direction: negative / neutral / positive) vs "magnitude" (strength)
- Average emotion strengths for 5 emotions (joy, sadness, anger, fear, disgust)
- Which words contribute most to the sentiment (positive and negative)
- Which words are top examples for each emotion
- Diagnostics for interpretability: coverage (how many tokens matched the lexicon),
  standard deviation of token scores (variability), and top "out-of-vocabulary" words

Design goals:
- Keep dependencies minimal (csv, re, collections, pathlib, unicodedata) so this file
  can run anywhere Python runs.
- Be fast across repeated calls by caching the lexicon in memory and reusing it unless
  the CSV file changes on disk (mtime check).
- Remain transparent and explainable: every aggregate is the sum/average of per-token
  values you can inspect in the "tokens" list in the return object.

NOTE on CSV structure (expected columns):
  word,sentiment_score,joy,sadness,anger,fear,disgust
The loader is defensive: missing/invalid numeric fields become 0.0.
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path          # Path handling that works on all OSes
import csv                        # CSV reading with DictReader for named columns
import re                         # Regular expressions for tokenization
import unicodedata                # Unicode normalization to clean fancy punctuation
from math import sqrt             # Only need sqrt for standard deviation
from collections import Counter, defaultdict   # Frequency counting and default 0.0 dict
from typing import Dict, List, Tuple           # Type hints for clarity (optional)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS / CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# Where the lexicon CSV lives. We compute it relative to THIS file so importing
# from other working directories still finds the CSV as long as it sits next to
# the script. This avoids surprises with cwd().
LEXICON_PATH = Path(__file__).resolve().parent / "sentiart_lexicon.csv"

# These emotion column names are the "canonical five" for our analyser. If you add
# more emotions to the CSV later, add the new headers here as well.
EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust"]

# Tokenization:
# - WORD_PATTERN tries to catch "word-like" sequences:
#   - must start with a letter or digit (so we don't capture stray punctuation)
#   - can continue with letters/digits/apostrophes/hyphens including curly quotes
#   - examples matched: "don't", "rock-n-roll", "O’Reilly", "naïve", "2024"
#   Why ASCII classes here and not full \p{L}? Standard 're' doesn't support it.
#   We instead normalize to NFKC so many fancy characters fold to basic forms,
#   and allow digits plus ASCII letters. This keeps dependency-free and robust.
WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’-]*")

# Simple ASCII emoticons (not full emoji handling) — we pre-extract these so that
# tokenization won't eat them. This small list is easy to expand. Each emoticon
# tends to carry clear sentiment and is common in casual text.
EMOTICON_PATTERN = re.compile(r"(:\)|:\(|:D|:P|;\)|:\/|:-\)|:-\(|:'\(|<3)")

# Common function words and contraction fragments to exclude from OOV display
STOPWORDS = {
    "a","an","and","the","of","to","in","on","for","with","at","by","from","as",
    "that","this","it","is","are","was","were","be","been","being","do","does","did",
    "but","or","if","then","so","than","very","can","could","should","would","will",
    "just","not","no","nor","you","your","yours","i","me","my","we","our","they","their",
    "he","she","his","her","them","who","whom","which","what","when","where","why","how",
    "also","into","over","under","again","more","most","such","only","own","same",
    # common contraction shards
    "s","t","d","ll","m","o","re","ve","y"
}


# ──────────────────────────────────────────────────────────────────────────────
# LEXICON CACHE (module-level singletons)
# ──────────────────────────────────────────────────────────────────────────────
# We keep the lexicon dictionary in memory after the first load, and also remember
# the file's modification time. If the file hasn't changed, we skip re-reading it.
# This is a simple, safe optimization for repeated calls (e.g. in a web server).
_LEXICON_CACHE: Dict[str, Dict[str, float]] = {}
_LEXICON_MTIME: float = -1.0


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: TEXT NORMALIZATION & TOKENIZATION
# ──────────────────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    Normalize Unicode text so tokens are consistent.

    - Ensures input is a string (str())
    - Applies NFKC normalization, which:
        * de-composes + re-composes characters into a canonical form
        * folds some "fancy" punctuation into simpler equivalents where applicable
      Why: this reduces weird cases like multiple forms of apostrophes or width variants.
    """
    if not isinstance(text, str):
        # If someone passes bytes, numbers, or some object with __str__, we convert.
        text = str(text)
    # Normalize to NFKC to unify visually-similar Unicode characters.
    text = unicodedata.normalize("NFKC", text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Convert raw text into a list of tokens suitable for lexicon lookup.

    What we do:
      1) Normalize Unicode (NFKC) so punctuation is consistent.
      2) Extract "emoticons" FIRST and surround them with spaces, so later regex
         doesn't swallow or split them weirdly. We then keep them as separate tokens.
      3) Find word-like tokens via WORD_PATTERN (letters/digits + optional ' or -).
      4) Casefold ALL tokens (casefold() is stronger than lower() for Unicode).
      5) Return combined list: words + emoticons.

    Note: The order of tokens returned is not strictly the original order — because
    we gather words then add emoticons — but for our aggregate statistics this is OK.
    If you need strict order (e.g., sentence position), collect indices during matching.
    """
    # 1) Normalize text
    text = normalize_text(text)

    # 2) Extract emoticons and pad with spaces so later token search won't break them.
    # We find all emoticons, then insert spaces around each match starting from the end
    # of the string so that earlier insertions don't change the indices of later spans.
    emoticons = list(EMOTICON_PATTERN.finditer(text))
    for m in reversed(emoticons):
        start, end = m.span()
        # Insert spaces around the emoticon: "...X:)" -> "...X :) "
        text = text[:start] + " " + text[start:end] + " " + text[end:]

    # 3) Extract "word" tokens (letters/digits with optional internal ' or -)
    words = WORD_PATTERN.findall(text)

    # 4) Keep emoticons as tokens (the matched string itself)
    emo_tokens = [m.group(0) for m in emoticons]

    # 5) Combine sets. For strict order, you'd sort by original indices; not required here.
    all_tokens = words + emo_tokens

    # 6) Casefold tokens to improve matching against lexicon entries regardless of case.
    # Casefold example: "Straße".lower() -> "straße", "Straße".casefold() -> "strasse"
    # which is often what you want for matching.
    return [t.casefold() for t in all_tokens]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: LEXICON LOADING (with caching and defensive parsing)
# ──────────────────────────────────────────────────────────────────────────────
def _parse_float(value: str) -> float:
    """
    Convert a CSV field to float, defaulting to 0.0 on any error or blank.

    Why: We don't want the analyser to crash on a malformed row; we prefer to
    degrade gracefully and keep going. Logging could be added if you want to
    count or inspect bad rows.
    """
    try:
        return float(value)
    except Exception:
        return 0.0


def load_lexicon() -> Dict[str, Dict[str, float]]:
    """
    Load the sentiment lexicon from CSV into a dictionary keyed by token.

    Returns:
      A dictionary of:
        {
          "word": {
             "sentiment_score": float,
             "joy": float,
             "sadness": float,
             "anger": float,
             "fear": float,
             "disgust": float
          },
          ...
        }

    Caching:
      - Uses module-level _LEXICON_CACHE and _LEXICON_MTIME.
      - If LEXICON_PATH's modification time hasn't changed since last load,
        we return the in-memory cache.
      - Otherwise, we parse the CSV and refresh the cache.

    Failure handling:
      - If the file isn't found, returns an empty dict (no crash).
    """
    global _LEXICON_CACHE, _LEXICON_MTIME

    path = LEXICON_PATH

    # Get file modification time; if the file is missing, return an empty lexicon.
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _LEXICON_CACHE = {}
        _LEXICON_MTIME = -1.0
        return _LEXICON_CACHE

    # If we have a cache and the file hasn't changed, reuse it.
    if _LEXICON_CACHE and _LEXICON_MTIME == mtime:
        return _LEXICON_CACHE

    # Otherwise, parse the CSV fresh.
    lexicon: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # DictReader yields each row as a dict keyed by column names.
        # Missing columns will yield None; _parse_float handles that.
        for row in reader:
            # Normalize and validate the "word" field
            word = (row.get("word") or "").strip()
            if not word:
                # Skip empty "word" entries rather than inserting blanks
                continue

            # Build the numeric entry. Missing/invalid values become 0.0, which is safe.
            entry = {
                "sentiment_score": _parse_float(row.get("sentiment_score", "0")),
                "joy": _parse_float(row.get("joy", "0")),
                "sadness": _parse_float(row.get("sadness", "0")),
                "anger": _parse_float(row.get("anger", "0")),
                "fear": _parse_float(row.get("fear", "0")),
                "disgust": _parse_float(row.get("disgust", "0")),
            }

            # Casefold the key so lookups are case-insensitive and Unicode-friendly.
            lexicon[word.casefold()] = entry

    # Refresh cache + mtime
    _LEXICON_CACHE = lexicon
    _LEXICON_MTIME = mtime
    return _LEXICON_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def _is_good_oov(token: str) -> bool:
    """
    Filter for display-worthy OOV tokens:
    - alphabetic only (drops numbers/punctuation fragments)
    - length >= 3 (drops very short noise)
    - not in STOPWORDS
    """
    w = token.lower()
    return w.isalpha() and len(w) >= 3 and w not in STOPWORDS

def analyze_text(text: str) -> dict:
    """
    Main entry point for callers.

    Args:
        text: any string-like object containing the text to analyse (str() is applied).

    Returns:
        A dictionary with four top-level keys:

        {
          "summary": {
             "sentiment_score_mean": float,   # average sentiment over matched tokens
             "polarity": -1|0|1,              # direction only (negative, neutral, positive)
             "magnitude": float,              # total strength regardless of sign
             "stddev": float,                 # variability of token sentiment scores
             "token_count": int,              # total tokens we saw
             "matched_token_count": int,      # tokens that had a lexicon entry
             "coverage": float,               # matched_token_count / token_count
             "positive_ratio": float,         # share of matched tokens with score > +0.05
             "negative_ratio": float,         # share with score < -0.05
             "neutral_ratio":  float,         # share neither > +0.05 nor < -0.05
             "oov_examples": [(str,int)],     # top 10 unknown words and their counts
             "lexicon_rows": int              # number of entries in the loaded lexicon
          },

          "emotions": {
             "joy": float, "sadness": float, "anger": float, "fear": float, "disgust": float
          },

          "top_contributors": {
             "positive": [token_rows...],     # top 10 by positive contribution (score * count)
             "negative": [token_rows...],     # top 10 by negative contribution
             "by_emotion": {
                "joy": [token_rows...], ...   # top 10 tokens per emotion by |emotion_score| * count
             }
          },

          "tokens": [
             {
               "word": str,
               "count": int,
               "sentiment_score": float,      # raw per-token score from lexicon
               "contribution": float,         # sentiment_score * count (signed)
               "emotions": {e: float for e in EMOTIONS}
             },
             ...
          ]
        }

    Notes:
      - A "token" here is a casefolded word-like string or one of our simple emoticons.
      - Only tokens that exist in the lexicon appear in "tokens"; unknown tokens are
        summarized in "oov_examples".
      - All averages are weighted by token count: frequent words move the needle more.
    """
    # Tokenize input. total_tokens is all tokens we saw; some may be OOV (unknown).
    tokens = tokenize(text)
    total_tokens = len(tokens)

    # Load (or reuse) the lexicon from CSV; caching makes this fast if repeated.
    lexicon = load_lexicon()

    # Frequency of each token in the document. Counter returns dict-like counts.
    freq = Counter(tokens)

    # We'll maintain two lists of (word, count) for matched vs unmatched tokens.
    matched_tokens: List[Tuple[str, int]] = []
    unmatched_tokens: List[Tuple[str, int]] = []

    # Aggregates for sentiment:
    total_sentiment = 0.0  # sum of (token_score * token_count)
    total_weight = 0       # total matched token instances (denominator for averages)

    # Per-emotion accumulators (weighted by token count).
    emotion_sums = defaultdict(float)

    # Per-token rows for transparency/debugging; each corresponds to one unique word.
    tokens_output: List[Dict] = []

    # Iterate unique tokens and their counts (not every occurrence).
    for word, count in freq.items():
        entry = lexicon.get(word)

        if entry is None:
            # OOV: no lexicon entry — we can't assign a sentiment score.
            unmatched_tokens.append((word, count))
            continue

        # Matched token: keep for ratios and aggregates
        matched_tokens.append((word, count))

        # Raw per-token score from lexicon (could be negative, zero, or positive).
        score = entry["sentiment_score"]

        # "Contribution" is how much THIS unique token contributes to the total.
        # If "good" appears 5 times with score +0.3, contribution = +1.5.
        # If "terrible" appears 2 times with score -0.9, contribution = -1.8.
        contribution = score * count

        # Add to running totals for average calculation and magnitude.
        total_sentiment += contribution
        total_weight += count

        # Emotion sums: also weighted by how often the word appears.
        for e in EMOTIONS:
            emotion_sums[e] += entry[e] * count

        # Record a detailed token row for explainability and "top" lists.
        tokens_output.append({
            "word": word,
            "count": count,
            "sentiment_score": score,
            "contribution": contribution,  # signed (positive or negative)
            "emotions": {e: entry[e] for e in EMOTIONS}
        })

    # Compute the mean sentiment score across matched tokens (weighted).
    # If there were no matches, avoid ZeroDivision by returning 0.0 (neutral).
    sentiment_score_mean = (total_sentiment / total_weight) if total_weight else 0.0

    # Ratios of positive / negative / neutral matched tokens.
    # We use small thresholds (±0.05) to avoid counting near-zero "noise" as signal.
    pos_tokens = sum(c for (w, c) in matched_tokens if lexicon[w]["sentiment_score"] > 0.05)
    neg_tokens = sum(c for (w, c) in matched_tokens if lexicon[w]["sentiment_score"] < -0.05)
    neu_tokens = total_weight - pos_tokens - neg_tokens

    # Convert these counts into ratios in [0,1]; again, guard for empty matched set.
    positive_ratio = (pos_tokens / total_weight) if total_weight else 0.0
    negative_ratio = (neg_tokens / total_weight) if total_weight else 0.0
    neutral_ratio  = (neu_tokens  / total_weight) if total_weight else 0.0

    # "Coverage" tells you how much of the text the lexicon could actually score.
    # 1.0 == everything matched; 0.0 == nothing matched. Helpful for confidence.
    coverage = (total_weight / total_tokens) if total_tokens else 0.0

    # "Magnitude" reflects overall intensity regardless of sign:
    # sum over tokens of |score| * count. This highlights strongly-opinionated texts
    # even if positives and negatives cancel in the mean.
    magnitude = sum(abs(t["contribution"]) for t in tokens_output)

    # Standard deviation (weighted) of token sentiment scores. A high stddev means
    # the document mixes strongly positive and strongly negative words; a low stddev
    # means the sentiment is more consistent. We compute variance as:
    #   sum(count * (score - mean)^2) / total_weight
    if total_weight:
        var = sum(t["count"] * ((t["sentiment_score"] - sentiment_score_mean) ** 2)
                  for t in tokens_output) / total_weight
        stddev = sqrt(var)
    else:
        stddev = 0.0

    # Per-emotion averages: divide weighted sums by total matched token instances.
    # If nothing matched, all emotion averages are 0.0.
    emotion_avgs = {e: (emotion_sums[e] / total_weight) if total_weight else 0.0
                    for e in EMOTIONS}

    # "Top contributors" by absolute signed contribution. This surfaces the words that
    # actually moved the final score the most due to both strength and frequency.
    tokens_output_sorted = sorted(tokens_output,
                                  key=lambda x: abs(x["contribution"]),
                                  reverse=True)
    # Positive movers: contributions > 0, take top 10
    top_positive = [t for t in tokens_output_sorted if t["contribution"] > 0][:10]
    # Negative movers: contributions < 0, take top 10
    top_negative = [t for t in tokens_output_sorted if t["contribution"] < 0][:10]

    # For "top by emotion", we rank by |emotion_score| * count to surface frequent
    # and strongly-emotive words for each emotion separately.
    top_by_emotion = {}
    for e in EMOTIONS:
        unique = sorted(
            tokens_output,
            key=lambda t: abs(t["emotions"][e]) * t["count"],
            reverse=True
        )[:10]
        top_by_emotion[e] = unique

    # OOV (out-of-vocabulary) examples: top 10 unknown words by frequency.
    # Useful for: expanding the lexicon, debugging domain-specific terms, etc.
    oov_filtered = [(w, c) for (w, c) in unmatched_tokens if _is_good_oov(w)]
    oov_examples = sorted(oov_filtered, key=lambda x: x[1], reverse=True)[:10]

    # "Polarity" collapses direction to {-1, 0, +1} based on the mean sentiment.
    # This is often handy for quick UI indicators (thumbs up/down/neutral).
    if sentiment_score_mean > 0.0:
        polarity = 1
    elif sentiment_score_mean < 0.0:
        polarity = -1
    else:
        polarity = 0

    # Return a single, self-contained dictionary so callers can serialize to JSON,
    # persist to a DB, or render in a UI without further work.
    return {
        "summary": {
            # Core aggregates
            "sentiment_score_mean": sentiment_score_mean,
            "polarity": polarity,             # -1 negative, 0 neutral, +1 positive
            "magnitude": magnitude,           # intensity regardless of sign
            "stddev": stddev,                 # variability of token sentiment

            # Diagnostics
            "token_count": total_tokens,
            "matched_token_count": total_weight,
            "coverage": coverage,

            # Composition of matched tokens by sign
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,

            # Visibility into unknown tokens + lexicon size (for sanity checks)
            "oov_examples": oov_examples,     # e.g., [("word1", 7), ("word2", 5), ...]
            "lexicon_rows": len(load_lexicon())
        },

        # Average emotion strengths (weighted by token counts)
        "emotions": emotion_avgs,

        # What moved the needle most (positive, negative), plus emotional exemplars
        "top_contributors": {
            "positive": top_positive,
            "negative": top_negative,
            "by_emotion": top_by_emotion
        },

        # Per-token transparency (only tokens present in lexicon)
        "tokens": tokens_output
    }
