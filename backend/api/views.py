from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
import json
import os
import re
from django.conf import settings
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from api.keyness.keyness_analyser import compute_keyness
from gensim import corpora, models
from collections import defaultdict
from api.keyness.keyness_analyser import compute_keyness, keyness_gensim, keyness_spacy
import spacy
import mimetypes
from django.core.files.uploadedfile import UploadedFile

CORPUS_DIR = os.path.join(settings.BASE_DIR, "api", "corpus")
SAMPLE_FILE = os.path.join(CORPUS_DIR, "sample1.txt")

# File validation constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_FILES = 5
ALLOWED_EXTENSIONS = {'.txt', '.doc', '.docx'}
ALLOWED_MIME_TYPES = {
    'text/plain',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}


def validate_file(uploaded_file: UploadedFile) -> tuple[bool, str]:
    """
    Validate a single uploaded file.
    Returns (is_valid, error_message)
    """
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"{uploaded_file.name} exceeds maximum file size (5MB)"

    # Check if file is empty
    if uploaded_file.size == 0:
        return False, f"{uploaded_file.name} is empty"

    # Check file extension
    file_extension = uploaded_file.name.lower().split('.')[-1] if '.' in uploaded_file.name else ''
    if f'.{file_extension}' not in ALLOWED_EXTENSIONS:
        return False, f"{uploaded_file.name} has an unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    # Check MIME type for additional security
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        return False, f"{uploaded_file.name} has an invalid MIME type"

    # Check for suspicious file names
    suspicious_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
    if any(char in uploaded_file.name for char in suspicious_chars):
        return False, f"{uploaded_file.name} contains invalid characters"

    return True, ""


def process_text_file(uploaded_file: UploadedFile) -> tuple[str, str]:
    """
    Process different file types and extract text content.
    Returns (text_content, error_message)
    """
    try:
        file_extension = uploaded_file.name.lower().split('.')[-1]

        if file_extension == 'txt':
            # Handle text files
            content = uploaded_file.read()

            # Try different encodings
            for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return "", "Unable to decode file with supported encodings"


        elif file_extension == 'docx':

            from docx import Document

            doc = Document(uploaded_file)

            text_content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

        elif file_extension == 'doc':

            return "", ".doc files not supported. Please convert to .docx or .txt"

        else:
            return "", f"Unsupported file type: {file_extension}"

        # Basic content validation
        if len(text_content.strip()) == 0:
            return "", "File appears to be empty or contains only whitespace"

        # Check for reasonable content length
        if len(text_content) > 1000000:  # 1MB of text
            return "", "Text content is too large"

        return text_content, ""

    except Exception as e:
        return "", f"Error processing file: {str(e)}"


@csrf_exempt
@require_POST
def upload_files(request):
    if not request.FILES:
        return JsonResponse({'error': 'No files uploaded'}, status=400)

    # Check number of files
    if len(request.FILES) > MAX_FILES:
        return JsonResponse({
            'error': f'Too many files. Maximum {MAX_FILES} files allowed'
        }, status=400)

    processed_files = []
    errors = []

    for field_name, uploaded_file in request.FILES.items():
        # Validate file
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            errors.append(error_msg)
            continue

        # Process file content
        text_content, process_error = process_text_file(uploaded_file)
        if process_error:
            errors.append(f"{uploaded_file.name}: {process_error}")
            continue

        processed_files.append({
            'filename': uploaded_file.name,
            'file_size': uploaded_file.size,
            'text_content': text_content,
            'word_count': len(text_content.split()) if text_content else 0,
            'char_count': len(text_content) if text_content else 0,
        })

    # Return appropriate response
    if not processed_files and errors:
        return JsonResponse({
            'success': False,
            'error': 'No files could be processed',
            'details': errors
        }, status=400)

    return JsonResponse({
        'success': True,
        'files': processed_files,
        'errors': errors,  # Include any non-fatal errors
        'total_files_processed': len(processed_files),
        'total_files_failed': len(errors)
    })

def read_corpus():
    try:
        with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def keyness_sklearn(uploaded_text, corpus_text, top_n=20):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([uploaded_text, corpus_text])
    terms = vectorizer.get_feature_names_out()
    freqs = X.toarray()

    uploaded_counts = freqs[0]
    corpus_counts = freqs[1]

    results = []
    for word, u_count, c_count in zip(terms, uploaded_counts, corpus_counts):
        if u_count == 0:
            continue  # skip words not in uploaded text

        contingency = np.array([
            [u_count, sum(uploaded_counts) - u_count],
            [c_count, sum(corpus_counts) - c_count]
        ])

        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        results.append({
            "word": word,
            "uploaded_count": int(u_count),
            "sample_count": int(c_count),
            "chi2": float(chi2),
            "p_value": float(p),
        })

    # Sort and limit top N
    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    results_top = results_sorted[:top_n]

    return {
        "results": results_top,
        "uploaded_total": int(sum(uploaded_counts)),
        "corpus_total": int(sum(corpus_counts)),
    }

def keyness_gensim(uploaded_text, corpus_text, top_n=20):
    """
    Compute keyness using Gensim TF-IDF scores and chi-squared contingency for counts.
    """
    # 1. Tokenize texts
    uploaded_tokens = uploaded_text.lower().split()
    corpus_tokens = corpus_text.lower().split()

    # 2. Build dictionary and corpus for gensim
    dictionary = corpora.Dictionary([uploaded_tokens, corpus_tokens])
    corpus_gensim = [dictionary.doc2bow(uploaded_tokens), dictionary.doc2bow(corpus_tokens)]

    # 3. TF-IDF model
    tfidf = models.TfidfModel(corpus_gensim, smartirs='ntc')
    tfidf_scores = [tfidf[doc] for doc in corpus_gensim]  # list of (term_id, score)

    # 4. Convert to dict for easy access
    uploaded_tfidf = {dictionary[id]: score for id, score in tfidf_scores[0]}
    corpus_tfidf = {dictionary[id]: score for id, score in tfidf_scores[1]}

    # 5. Count occurrences
    uploaded_counts = defaultdict(int)
    corpus_counts = defaultdict(int)
    for word in uploaded_tokens:
        uploaded_counts[word] += 1
    for word in corpus_tokens:
        corpus_counts[word] += 1

    # 6. Compute chi-squared contingency and combine with TF-IDF
    results = []
    all_words = set(uploaded_tokens)
    for word in all_words:
        u_count = uploaded_counts.get(word, 0)
        c_count = corpus_counts.get(word, 0)
        if u_count == 0:
            continue

        contingency = np.array([
            [u_count, sum(uploaded_counts.values()) - u_count],
            [c_count, sum(corpus_counts.values()) - c_count]
        ])
        try:
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except ValueError:
            chi2, p = 0, 1

        results.append({
            "word": word,
            "uploaded_count": u_count,
            "sample_count": c_count,
            "chi2": float(chi2),
            "p_value": float(p),
            "tfidf_score": float(uploaded_tfidf.get(word, 0))
        })

    # Sort by chi2 or by TF-IDF (here by chi2 for compatibility)
    results_sorted = sorted(results, key=lambda x: x["chi2"], reverse=True)
    results_top = results_sorted[:top_n]

    return {
        "results": results_top,
        "uploaded_total": int(sum(uploaded_counts.values())),
        "corpus_total": int(sum(corpus_counts.values())),
    }

@csrf_exempt
@require_http_methods(["GET"])
def get_corpus_preview(request):
        try:
            corpus_text = read_corpus()
            if not corpus_text:
                return JsonResponse({"error": "No corpus files found."}, status=404)

            lines = [line.strip() for line in corpus_text.split("\n") if line.strip()][:4]
            preview = "\n".join(lines)
            return JsonResponse({"preview": preview})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def analyse_keyness(request):
    """
    Perform keyness analysis using the requested method.
    POST body: { "uploaded_text": "...", "method": "nltk" }
    """
    try:
        data = json.loads(request.body)
        uploaded_text = data.get("uploaded_text", "")
        method = data.get("method", "nltk").lower()  # default to nltk

        if not uploaded_text:
            return JsonResponse({"error": "No uploaded text provided"}, status=400)

        sample_corpus = read_corpus()
        if not sample_corpus:
            return JsonResponse({"error": "Reference corpus is empty."}, status=500)

        if method == "nltk":
            from api.keyness.keyness_analyser import compute_keyness
            results_list = compute_keyness(uploaded_text, sample_corpus, top_n=20)
            results = {
                "results": results_list,
                "uploaded_total": len(uploaded_text.split()),
                "corpus_total": len(sample_corpus.split())
            }

        elif method == "sklearn":
            results = keyness_sklearn(uploaded_text, sample_corpus)

        elif method == "gensim":
            from api.keyness.keyness_analyser import keyness_gensim
            results = keyness_gensim(uploaded_text, sample_corpus)

        elif method.lower() == "spacy":
            results = keyness_spacy(uploaded_text, sample_corpus)


        else:
            return JsonResponse({"error": f"Unknown method: {method}"}, status=400)

        response = {
            "method": method,
            "results": results["results"],
            "uploaded_total": results.get("uploaded_total", len(uploaded_text.split())),
            "corpus_total": results.get("corpus_total", len(sample_corpus.split())),
            "preview": "\n".join([line for line in sample_corpus.splitlines() if line.strip()][:4])
        }

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
