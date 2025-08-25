# api/utils.py
import os
import re
import math
from collections import Counter
import docx
import tempfile
import mammoth
from django.conf import settings

CORPUS_DIR = os.path.join(settings.BASE_DIR, 'api', 'corpus')


# ---------------------- Text Utilities ---------------------- #
def tokenize_and_count(text):
    """Tokenize text and return word frequencies"""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    words = [w for w in words if len(w) > 2]
    freq = Counter(words)
    return {
        "words": words,
        "freq": dict(freq),
        "total": len(words)
    }


def calculate_log_likelihood(a, b, c, d):
    """Calculate log-likelihood ratio"""
    if a == 0 or b == 0:
        return 0
    e1 = c * (a + b) / (c + d)
    e2 = d * (a + b) / (c + d)
    try:
        ll = 2 * ((a * math.log(a / e1)) + (b * math.log(b / e2)))
        return ll if not math.isnan(ll) else 0
    except (ValueError, ZeroDivisionError):
        return 0


def perform_keyness_analysis(uploaded_text, sample_text):
    """Perform keyness analysis between two texts"""
    uploaded_data = tokenize_and_count(uploaded_text)
    sample_data = tokenize_and_count(sample_text)
    all_words = set(list(uploaded_data["freq"].keys()) + list(sample_data["freq"].keys()))
    results = []

    for word in all_words:
        uploaded_count = uploaded_data["freq"].get(word, 0)
        sample_count = sample_data["freq"].get(word, 0)
        if uploaded_count + sample_count < 2:
            continue

        uploaded_other = uploaded_data["total"] - uploaded_count
        sample_other = sample_data["total"] - sample_count
        ll = calculate_log_likelihood(uploaded_count, sample_count, uploaded_other, sample_other)

        if ll > 3.84:  # significant threshold
            uploaded_freq = (uploaded_count / uploaded_data["total"]) * 1000
            sample_freq = (sample_count / sample_data["total"]) * 1000
            effect_size = uploaded_freq / (sample_freq + 0.001)
            results.append({
                "word": word,
                "uploaded_count": uploaded_count,
                "sample_count": sample_count,
                "uploaded_freq": round(uploaded_freq, 2),
                "sample_freq": round(sample_freq, 2),
                "log_likelihood": round(ll, 2),
                "effect_size": round(effect_size, 2),
                "keyness": "Positive" if uploaded_count > sample_count else "Negative"
            })

    results.sort(key=lambda x: x["log_likelihood"], reverse=True)
    return {
        "results": results,
        "uploaded_total": uploaded_data["total"],
        "sample_total": sample_data["total"]
    }


# ---------------------- Corpus Utilities ---------------------- #
def read_corpus():
    """Read all corpus files and return combined text"""
    corpus_text = ""
    try:
        for filename in os.listdir(CORPUS_DIR):
            if filename.endswith(".txt"):
                with open(os.path.join(CORPUS_DIR, filename), "r", encoding="utf-8") as f:
                    corpus_text += f.read() + " "
        return corpus_text
    except FileNotFoundError:
        return ""


# ---------------------- Document Parsing ---------------------- #
def extract_text_from_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    text = ""

    # TXT
    if file_name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    # DOCX
    elif file_name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            for chunk in uploaded_file.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        try:
            doc = docx.Document(tmp_file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        finally:
            os.unlink(tmp_file_path)

    # DOC
    elif file_name.endswith(".doc"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp_file:
            for chunk in uploaded_file.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        try:
            with open(tmp_file_path, "rb") as doc_file:
                result = mammoth.extract_raw_text(doc_file)
                text = result.value
        finally:
            os.unlink(tmp_file_path)

    else:
        raise ValueError("Unsupported file format")

    # Preview
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    preview = "\n".join(lines[:4])

    return {
        "text": text,
        "preview": preview,
        "word_count": len(text.split())
    }
