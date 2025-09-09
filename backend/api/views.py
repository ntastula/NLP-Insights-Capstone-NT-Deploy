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
from gensim import corpora, models
from collections import defaultdict
from api.keyness.keyness_analyser import (
    compute_keyness,
    keyness_gensim,
    keyness_spacy,
    keyness_sklearn,
    filter_content_words,
    filter_all_words,
)
import spacy
import mimetypes
from django.core.files.uploadedfile import UploadedFile
from .models import KeynessResult
import logging

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

logger = logging.getLogger('api')

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
        logger.warning("Upload attempt with no files")
        return JsonResponse({'error': 'No files uploaded'}, status=400)

    if len(request.FILES) > MAX_FILES:
        logger.warning(f"Upload attempt with too many files: {len(request.FILES)}")
        return JsonResponse({
            'error': f'Too many files. Maximum {MAX_FILES} files allowed'
        }, status=400)

    processed_files = []
    errors = []

    for uploaded_file in request.FILES.getlist("files"):
        is_valid, error_msg = validate_file(uploaded_file)
        if not is_valid:
            logger.warning(f"File validation failed: {uploaded_file.name} - {error_msg}")
            errors.append(error_msg)
            continue

        text_content, process_error = process_text_file(uploaded_file)
        if process_error:
            logger.warning(f"File processing error: {uploaded_file.name} - {process_error}")
            errors.append(f"{uploaded_file.name}: {process_error}")
            continue

        logger.info(
            f"File uploaded: {uploaded_file.name} "
            f"({uploaded_file.size} bytes, {len(text_content.split())} words)"
        )
        processed_files.append({
            "filename": uploaded_file.name,
            "file_size": uploaded_file.size,
            "text_content": text_content,
            "word_count": len(text_content.split()) if text_content else 0,
            "char_count": len(text_content) if text_content else 0,
        })

    logger.info(f"Upload complete. {len(processed_files)} files processed, {len(errors)} failed.")
    return JsonResponse({
        'success': True,
        'files': processed_files,
        'errors': errors,
        'total_files_processed': len(processed_files),
        'total_files_failed': len(errors)
    })

def read_corpus():
    try:
        with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""





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
    try:
        data = json.loads(request.body)
        uploaded_text = data.get("uploaded_text", "")
        method = data.get("method", "nltk").lower()
        filter_mode = data.get("filter_mode", "content")

        if filter_mode == "all":
            filter_fn = filter_all_words
        else:
            filter_fn = filter_content_words

        if not uploaded_text:
            logger.warning("Analysis request with empty uploaded_text")
            return JsonResponse({"error": "No uploaded text provided"}, status=400)

        sample_corpus = read_corpus()
        if not sample_corpus:
            logger.error("Reference corpus is empty")
            return JsonResponse({"error": "Reference corpus is empty."}, status=500)

        # Choose filter function
        filter_func = filter_all_words if filter_mode == "all" else filter_content_words

        # Apply filter consistently
        filtered_uploaded = filter_func(uploaded_text)
        filtered_corpus = filter_func(sample_corpus)

        logger.info(
            f"Keyness analysis requested: method={method}, "
            f"filter_mode={filter_mode}, "
            f"uploaded_text_words={len(filtered_uploaded)}, "
            f"corpus_words={len(filtered_corpus)}"
        )

        # Compute results
        if method == "nltk":
            results_list = compute_keyness(uploaded_text, sample_corpus, top_n=20, filter_func=filter_func)
        elif method == "sklearn":
            results_dict = keyness_sklearn(uploaded_text, sample_corpus, top_n=20, filter_func=filter_func)
            results_list = results_dict["results"]
        elif method == "gensim":
            results_dict = keyness_gensim(uploaded_text, sample_corpus, top_n=20, filter_func=filter_func)
            results_list = results_dict["results"]
        elif method == "spacy":
            results_dict = keyness_spacy(uploaded_text, sample_corpus, top_n=20, filter_func=filter_func)
            results_list = results_dict["results"]
        else:
            logger.warning(f"Unknown keyness method requested: {method}")
            return JsonResponse({"error": f"Unknown method: {method}"}, status=400)

        # ✅ Use filtered totals consistently
        uploaded_total = len(filtered_uploaded)
        corpus_total = len(filtered_corpus)

        # Save result to DB
        keyness_obj = KeynessResult.objects.create(
            method=method,
            uploaded_text=uploaded_text,
            results={"results": results_list},
            uploaded_total=uploaded_total,
            corpus_total=corpus_total,
        )

        logger.info(
            f"Keyness analysis completed: method={method}, "
            f"filter_mode={filter_mode}, "
            f"results_count={len(results_list)}, "
            f"uploaded_total={uploaded_total}, "
            f"corpus_total={corpus_total}, "
            f"id={keyness_obj.id}"
        )

        response = {
            "id": keyness_obj.id,
            "method": method,
            "filter_mode": filter_mode,
            "results": results_list,
            "uploaded_total": uploaded_total,
            "corpus_total": corpus_total,
        }

        return JsonResponse(response)

    except Exception as e:
        logger.exception(f"Error during keyness analysis: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def get_keyness_results(request, result_id):
    try:
            keyness_obj = KeynessResult.objects.get(id=result_id)
            response = {
                "id": keyness_obj.id,
                "method": keyness_obj.method,
                "results": keyness_obj.results_json.get("results", []),
                "uploaded_total": keyness_obj.uploaded_total,
                "corpus_total": keyness_obj.corpus_total,
                "created_at": keyness_obj.created_at.isoformat()
            }
            return JsonResponse(response)
    except KeynessResult.DoesNotExist:
            return JsonResponse({"error": "Result not found"}, status=404)


# --- Sentiment (SentiArt lexicon) -------------------------------------------

# Allow this view to accept POST requests
@csrf_exempt
# Only allow POST method (not GET, PUT, etc.)
@require_http_methods(["POST"])
def analyse_sentiment(request):

    # Try to load the JSON body of the request
    try:
        # request.body is raw bytes -> decode to string -> load as JSON
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        # If JSON is missing or invalid, return an error with status code 400 (bad request)
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    # Pull the uploaded_text field out of the JSON, or use empty string if missing
    text = (data.get("uploaded_text") or "").strip()

    # If no text was provided, return an error
    if not text:
        return JsonResponse({"error": "No uploaded_text provided"}, status=400)

    # Try to run the sentiment analyser
    try:
        # Import the function from our helper file (sentiart_analyser.py)
        from api.sentiment.sentiment_analyser import analyze_text

        # Call the analyser, passing the text
        result = analyze_text(text)

        # Return the result as JSON to the client
        return JsonResponse(result)

    # If the CSV lexicon file is missing, give a clear hint to the user
    except FileNotFoundError as e:
        return JsonResponse({
            "error": str(e),
            "hint": "Place CSV at backend/api/sentiment/sentiart_lexicon.csv "
                    "with headers: word,sentiment score,joy,sadness,anger,fear,disgust"
        }, status=500)

    # Catch any other errors and return them as JSON
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)