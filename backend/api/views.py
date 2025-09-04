from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
import json
import os, tempfile, shutil
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
from backend.utils.session_utils import SessionManager
from .models import UserUploadedFile
import logging
import traceback
from django.utils import timezone

CORPUS_DIR = os.path.join(settings.BASE_DIR, "api", "corpus")
SAMPLE_FILE = os.path.join(settings.BASE_DIR, 'api', 'corpus', 'sample1.txt')
logger = logging.getLogger(__name__)

# File validation constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_FILES = 5
ALLOWED_EXTENSIONS = {'.txt', '.doc', '.docx'}
ALLOWED_MIME_TYPES = {
    'text/plain',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

def read_corpus():
    try:
        with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

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
    try:
        logger.info(f"Session key: {request.session.session_key}")
        logger.info(f"Request.FILES: {request.FILES}")  # 🔹 show uploaded files
        logger.info(f"Request.POST: {request.POST}")

        if not request.FILES:
            logger.warning("No files received!")
            return JsonResponse({'error': 'No files uploaded'}, status=400)

        MAX_FILES = 5
        if len(request.FILES) > MAX_FILES:
            return JsonResponse({
                'error': f'Too many files. Maximum {MAX_FILES} files allowed'
            }, status=400)

        processed_files = []
        errors = []

        # Ensure session folder exists
        session_id = request.session.session_key
        if not session_id:
            request.session.create()
            session_id = request.session.session_key

        session_dir = os.path.join(tempfile.gettempdir(), f"uploads_{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        uploaded_files = request.FILES.getlist("file")  # get all files under the same key
        if not uploaded_files:
            return JsonResponse({'error': 'No files uploaded'}, status=400)

        for uploaded_file in uploaded_files:
            try:
                # existing validation & processing
                is_valid, error_msg = validate_file(uploaded_file)
                if not is_valid:
                    errors.append(error_msg)
                    continue

                text_content, process_error = process_text_file(uploaded_file)
                if process_error:
                    errors.append(f"{uploaded_file.name}: {process_error}")
                    continue

                user_file = SessionManager.save_user_file(
                    request=request,
                    filename=f"upload_{timezone.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}",
                    original_filename=uploaded_file.name,
                    file_size=uploaded_file.size,
                    file_type=uploaded_file.name.split('.')[-1].lower(),
                    text_content=text_content
                )

                processed_files.append({
                    'id': str(user_file.id),
                    'filename': uploaded_file.name,
                    'backend_filename': user_file.filename,
                    'file_size': user_file.file_size,
                    'text_content': user_file.text_content,
                    'word_count': user_file.word_count,
                    'char_count': user_file.char_count,
                })

            except Exception as e:
                errors.append(f"Error processing {uploaded_file.name}: {str(e)}")

        if not processed_files and errors:
            return JsonResponse({
                'success': False,
                'error': 'No files could be processed',
                'details': errors
            }, status=400)

        return JsonResponse({
            'success': True,
            'files': processed_files,
            'errors': errors,
            'total_files_processed': len(processed_files),
            'total_files_failed': len(errors)
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Server error: {str(e)}'
        }, status=500)


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


@csrf_exempt
@require_http_methods(["GET"])
def get_corpus_preview(request):
    """Get static corpus preview from corpus folder"""
    try:
        corpus_text = read_corpus()
        if not corpus_text:
            return JsonResponse({
                'success': True,
                'preview': 'Corpus file not found',
                'source': 'static_corpus'
            })

        lines = [line.strip() for line in corpus_text.split("\n") if line.strip()][:4]
        preview = "\n".join(lines)

        return JsonResponse({
            'success': True,
            'preview': preview,
            'source': 'static_corpus'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error getting corpus preview: {str(e)}'
        }, status=500)

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



@csrf_exempt
@require_http_methods(["GET"])
def get_user_files(request):
    """Get all files for the current user's session"""
    try:
        user_files = SessionManager.get_user_files(request)

        files_data = []
        for file_obj in user_files:
            files_data.append({
                'id': str(file_obj.id),
                'filename': file_obj.original_filename,
                'file_size': file_obj.file_size,
                'file_type': file_obj.file_type,
                'word_count': file_obj.word_count,
                'char_count': file_obj.char_count,
                'uploaded_at': file_obj.uploaded_at.isoformat(),
                'text_preview': file_obj.text_content[:200] + '...' if len(
                    file_obj.text_content) > 200 else file_obj.text_content
            })

        return JsonResponse({
            'success': True,
            'files': files_data,
            'total_files': len(files_data)
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error retrieving files: {str(e)}'
        }, status=500)


@csrf_exempt
@require_POST
def delete_user_file(request):
    """Delete a specific file for the current user"""
    try:
        data = json.loads(request.body)
        file_id = data.get('file_id')

        if not file_id:
            return JsonResponse({'error': 'File ID required'}, status=400)

        success = SessionManager.delete_user_file(request, file_id)

        if success:
            return JsonResponse({'success': True, 'message': 'File deleted successfully'})
        else:
            return JsonResponse({'error': 'File not found'}, status=404)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error deleting file: {str(e)}'
        }, status=500)


@csrf_exempt
@require_POST
def clear_user_data(request):
    """Clear all data for the current user's session"""
    try:
        SessionManager.clear_user_data(request)
        return JsonResponse({'success': True, 'message': 'All data cleared successfully'})
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error clearing data: {str(e)}'
        }, status=500)



@csrf_exempt
@require_POST
def save_analysis(request):
    """Save analysis results with session isolation"""
    try:
        data = json.loads(request.body)
        analysis_type = data.get('analysis_type')
        input_text = data.get('input_text', '')
        results = data.get('results', {})

        if not analysis_type or not results:
            return JsonResponse({'error': 'Analysis type and results required'}, status=400)

        analysis = SessionManager.save_analysis_result(
            request=request,
            analysis_type=analysis_type,
            input_text_preview=input_text,
            results=results
        )

        return JsonResponse({
            'success': True,
            'analysis_id': str(analysis.id),
            'message': 'Analysis saved successfully'
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error saving analysis: {str(e)}'
        }, status=500)


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