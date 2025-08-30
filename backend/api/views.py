from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
import json
import os
import re
import logging
import magic
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
from .utils import RateLimiter
from .exceptions import FileValidationError, RateLimitExceeded, ContentProcessingError
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.decorators import login_required


CORPUS_DIR = os.path.join(settings.BASE_DIR, "api", "corpus")
SAMPLE_FILE = os.path.join(CORPUS_DIR, "sample1.txt")
logger = logging.getLogger(__name__)

# Initialize rate limiter
upload_rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
analysis_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

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

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0]
    return request.META.get('REMOTE_ADDR')

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def secure_file_upload(request):
    """Handle secure file upload - supports multiple files"""
    try:
        client_ip = get_client_ip(request)

        # Rate limiting check
        if not upload_rate_limiter.is_allowed(client_ip):
            return JsonResponse({
                'error': 'Rate limit exceeded. Please wait before uploading again.'
            }, status=429)

        # Check if files were uploaded
        if not request.FILES:
            return JsonResponse({'error': 'No files uploaded'}, status=400)

        processed_files = []
        errors = []

        # Process each uploaded file
        for field_name, uploaded_file in request.FILES.items():
            try:
                result = FileProcessingService.process_uploaded_file(uploaded_file)
                processed_files.append(result)

                logger.info(f"Successfully processed: {result['filename']}, "
                            f"size: {result['file_size']}")

            except (FileValidationError, ContentProcessingError) as e:
                error_msg = f"Error processing {uploaded_file.name}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

        # Return results
        if not processed_files and errors:
            return JsonResponse({
                'error': 'No files could be processed',
                'details': errors
            }, status=400)

        response_data = {
            'success': True,
            'message': f'Successfully processed {len(processed_files)} file(s)',
            'files': processed_files
        }

        if errors:
            response_data['warnings'] = errors

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in file upload: {str(e)}")
        return JsonResponse({
            'error': 'An unexpected error occurred while processing your files'
        }, status=500)


@login_required
@require_http_methods(["POST"])
def analyse_keyness_secure(request):
    """Enhanced keyness analysis endpoint"""
    try:
        client_ip = get_client_ip(request)

        # Rate limiting
        if not analysis_rate_limiter.is_allowed(client_ip):
            return JsonResponse({
                'error': 'Rate limit exceeded. Please wait before analyzing again.'
            }, status=429)

        data = json.loads(request.body)

        # Validate input using our validators
        uploaded_text = data.get('uploaded_text', '').strip()
        method = data.get('method', '').lower()

        # Use TextValidator for validation
        TextValidator.validate_text_content(uploaded_text)

        # Validate method
        allowed_methods = ['nltk', 'sklearn', 'gensim', 'spacy']
        if method not in allowed_methods:
            return JsonResponse({
                'error': f'Invalid method. Allowed: {", ".join(allowed_methods)}'
            }, status=400)

        # Your existing analysis logic here...
        # This would call your actual keyness analysis functions

        # Example response (replace with your actual analysis)
        results = {
            'results': [
                {'word': 'example', 'score': 1.5, 'frequency': 10},
                # ... your actual results
            ],
            'uploaded_total': len(uploaded_text.split()),
            'corpus_total': 50000,
            'method_used': method
        }

        return JsonResponse(results)

    except FileValidationError as e:
        return JsonResponse({'error': str(e)}, status=400)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request'}, status=400)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return JsonResponse({'error': 'Analysis failed due to server error'}, status=500)


ALLOWED_TYPES = ['text/plain',
                 'application/msword',
                 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@csrf_exempt
@require_POST
def upload_files(request):
    if not request.FILES:
        return JsonResponse({'error': 'No files uploaded'}, status=400)

    processed_files = []
    errors = []

    for field_name, uploaded_file in request.FILES.items():
        if uploaded_file.size > 5 * 1024 * 1024:  # Max 5MB
            errors.append(f"{uploaded_file.name} exceeds maximum file size")
            continue
        if not uploaded_file.name.endswith(('.txt', '.doc', '.docx')):
            errors.append(f"{uploaded_file.name} is not an accepted file type")
            continue

        try:
            text_content = uploaded_file.read().decode('utf-8', errors='ignore')
            processed_files.append({
                'filename': uploaded_file.name,
                'file_size': uploaded_file.size,
                'text_content': text_content,
            })
        except Exception as e:
            errors.append(f"Error reading {uploaded_file.name}: {str(e)}")

    return JsonResponse({
        'success': True if processed_files else False,
        'files': processed_files,
        'errors': errors
    })