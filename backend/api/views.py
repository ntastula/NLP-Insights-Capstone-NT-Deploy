import gc
from django.http import JsonResponse, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
from django.views.decorators.http import require_GET
from rest_framework.decorators import api_view
from rest_framework.response import Response
from backend.utils.session_utils import ensure_session_exists, schedule_session_cleanup
from openai import OpenAI
import json
import os
import re
import requests
from django.conf import settings
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2  # <-- added chi2 for p-values
from gensim import corpora, models
from collections import defaultdict, Counter      # <-- added Counter
from api.keyness.keyness_analyser import (
    keyness_gensim,
    keyness_spacy,
    keyness_sklearn,
    filter_content_words,
    filter_all_words,
    extract_sentences,
    keyness_nltk,
)
import spacy
import mimetypes
import time
from django.core.files.uploadedfile import UploadedFile
from .models import KeynessResult
import logging
from pathlib import Path  # <-- ADDED (minimal import required)
import math               # <-- used by compute_keyness_from_counts

MAX_TEXT_LENGTH = 100000
logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def log_memory_usage(label):
    """Log current memory usage."""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory at {label}: {memory_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not log memory: {e}")

# ---- LLM generation helper (Ollama or Hugging Face) ----

def generate_text_with_fallback(prompt: str, num_predict: int = 600, temperature: float = 0.7) -> str:
    """
    Generate text using Groq, HuggingFace, or Ollama.
    Priority: Groq > HuggingFace > Ollama
    """
    log_memory_usage("LLM request start")

    provider = (os.environ.get("LLM_PROVIDER") or "groq").strip().lower()

    try:
        if provider == "groq":
            result = _generate_groq(prompt, num_predict, temperature)
        elif provider == "huggingface":
            result = _generate_huggingface(prompt, num_predict, temperature)
        else:  # ollama
            result = _generate_ollama(prompt, num_predict, temperature)

        # Clear memory after generation
        gc.collect()
        log_memory_usage("LLM request end")

        return result

    except Exception as e:
        logger.error(f"LLM generation failed with {provider}: {e}")
        gc.collect()
        raise


def _generate_groq(prompt: str, num_predict: int, temperature: float) -> str:
    """Generate text using Groq API (fast and free)."""
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Get one free at https://console.groq.com/keys")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    # Groq has a context limit, so truncate if needed
    max_prompt_length = 6000
    if len(prompt) > max_prompt_length:
        logger.warning(f"Prompt truncated from {len(prompt)} to {max_prompt_length} chars")
        prompt = prompt[:max_prompt_length]

    # Use Llama 3.1 8B - fast and high quality
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert data analyst and computational linguist. Provide clear, concise analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": min(num_predict, 1000),
        "temperature": temperature,
        "top_p": 0.9
    }

    logger.info("Calling Groq API with llama-3.1-8b-instant")

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code == 401:
        raise ValueError("Invalid Groq API key. Check your GROQ_API_KEY environment variable.")
    elif response.status_code == 429:
        raise ValueError("Groq rate limit exceeded. Please try again in a moment.")

    response.raise_for_status()

    data = response.json()

    # Extract the generated text
    if "choices" in data and len(data["choices"]) > 0:
        content = data["choices"][0].get("message", {}).get("content", "")
        if content:
            return content.strip()

    raise ValueError("Unexpected response format from Groq API")


def _generate_huggingface(prompt: str, num_predict: int, temperature: float) -> str:
    """Generate text using Hugging Face API."""
    hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    model = os.environ.get("HUGGINGFACE_MODEL") or "gpt2"

    if not hf_token:
        raise ValueError("HUGGINGFACE_API_TOKEN is not set")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    max_prompt_length = 1500
    if len(prompt) > max_prompt_length:
        logger.warning(f"Prompt truncated from {len(prompt)} to {max_prompt_length} chars")
        prompt = prompt[:max_prompt_length]

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": min(num_predict, 300),
            "temperature": temperature,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": True
        }
    }

    logger.info(f"Calling Hugging Face API with model: {model}")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 503:
                wait_time = (attempt + 1) * 10
                logger.warning(f"Model loading, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue

            if response.status_code == 200:
                break

            response.raise_for_status()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"Request timeout, retrying (attempt {attempt + 1}/{max_retries})")
                time.sleep(5)
                continue
            raise

    data = response.json()

    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict):
            text = data[0].get("generated_text", "")
            if text:
                return text.strip()

    if isinstance(data, dict):
        if "error" in data:
            raise ValueError(f"Hugging Face API Error: {data['error']}")
        text = data.get("generated_text", "")
        if text:
            return text.strip()

    raise ValueError("Failed to parse response from Hugging Face API")


def _generate_ollama(prompt: str, num_predict: int, temperature: float) -> str:
    """Generate text using Ollama local API."""
    base_url = os.environ.get("OLLAMA_URL") or "http://localhost:11434/api/generate"
    model = os.environ.get("OLLAMA_MODEL") or "llama3"

    max_prompt_length = 4000
    if len(prompt) > max_prompt_length:
        logger.warning(f"Prompt truncated from {len(prompt)} to {max_prompt_length} chars")
        prompt = prompt[:max_prompt_length]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": num_predict,
        }
    }

    logger.info(f"Calling Ollama API: {model}")
    response = requests.post(base_url, json=payload, timeout=180)
    response.raise_for_status()

    return (response.json() or {}).get("response", "")


@api_view(['GET'])
def test_groq(request):
    """Test endpoint to verify Groq API connection."""
    groq_api_key = os.environ.get("GROQ_API_KEY", "NOT_SET")

    logger.info("Testing Groq API connection...")

    if groq_api_key == "NOT_SET":
        return Response({
            'error': 'GROQ_API_KEY not set',
            'help': 'Get a free API key at https://console.groq.com/keys'
        }, status=500)

    # Test the API
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": "Say 'Hello, Groq is working!' in exactly those words."}
        ],
        "max_tokens": 20
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return Response({
                'status': 'success',
                'message': message,
                'model': 'llama-3.1-8b-instant',
                'groq_working': True
            })
        else:
            return Response({
                'status': 'error',
                'status_code': response.status_code,
                'response': response.text[:500]
            }, status=response.status_code)

    except Exception as e:
        return Response({
            'error': str(e),
            'groq_working': False
        }, status=500)


CORPUS_DIR = os.path.join(settings.BASE_DIR, "api", "corpus")
SAMPLE_FILE = os.path.join(CORPUS_DIR, "sample1.txt")
logger = logging.getLogger(__name__)

# --- Genre Corpus Meta (helpers only; no other behavior changed) -----------
# Looks for metadata-only corpora in backend/api/corpus_meta/*.json
META_DIR = Path("api/corpus_meta")
KEYNESS_DIR = Path("api/corpus_meta_keyness")


@api_view(['GET'])
def test_huggingface(request):
    """Test endpoint to debug HuggingFace API connection."""
    import requests

    hf_token = os.environ.get("HUGGINGFACE_API_TOKEN", "NOT_SET")
    hf_model = os.environ.get("HUGGINGFACE_MODEL", "gpt2")

    logger.info("Testing HuggingFace API connection...")
    logger.info(f"Token present: {hf_token != 'NOT_SET'}")
    logger.info(f"Token length: {len(hf_token) if hf_token != 'NOT_SET' else 0}")
    logger.info(f"Token starts with 'hf_': {hf_token.startswith('hf_') if hf_token != 'NOT_SET' else False}")
    logger.info(f"Model: {hf_model}")

    if hf_token == "NOT_SET":
        return Response({
            'error': 'HUGGINGFACE_API_TOKEN not set',
            'model': hf_model
        }, status=500)

    # Test the API
    url = f"https://api-inference.huggingface.co/models/{hf_model}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": "Hello, I am",
        "parameters": {
            "max_new_tokens": 10
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        return Response({
            'status_code': response.status_code,
            'model': hf_model,
            'url': url,
            'token_length': len(hf_token),
            'token_starts_correctly': hf_token.startswith('hf_'),
            'response': response.text[:500]
        })
    except Exception as e:
        return Response({
            'error': str(e),
            'model': hf_model,
            'token_length': len(hf_token)
        }, status=500)

def list_corpus_files(analysis_type=None):
    if analysis_type == "keyness":
        folder = KEYNESS_DIR
    else:
        folder = META_DIR

    if not folder.exists():
        return []

    files = [f.stem for f in folder.glob("*.json")]  # <- use stem (no .json)

    if analysis_type == "keyness":
        # strip _keyness suffix
        files = [f.replace("_keyness", "") for f in files]
        files.sort(key=lambda x: (0 if x == "general_fiction" else 1, x))
    else:
        files.sort()

    return files



def load_corpus_meta(corpus_name: str) -> dict:
    """
    Load a genre metadata JSON by name.
    Raises FileNotFoundError if the JSON does not exist.
    """
    p = Path(META_DIR) / f"{corpus_name}.json"
    if not p.exists():
        raise FileNotFoundError(f"Genre corpus meta not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def corpus_counts(meta: dict) -> tuple[dict, int]:
    """
    Extract (word -> count) and total_tokens from a loaded meta dict.
    Falls back to sum(freq.values()) if total_tokens is missing.
    """
    freq = meta.get("freq", {}) or {}
    total = meta.get("total_tokens")
    if total is None:
        try:
            total = sum(int(v) for v in freq.values())
        except Exception:
            total = 0
    return freq, int(total)

@csrf_exempt
@require_GET
def list_corpora(request):
    try:
        analysis_type = request.GET.get("analysis")
        files = list_corpus_files(analysis_type)

        # Log what was found
        logger.info("list_corpora -> analysis_type=%s, files=%s", analysis_type, files)

        # If keyness, add the suffix back for the frontend
        if analysis_type == "keyness":
            files = [f"{f}_keyness" if f != "general_fiction" else f"{f}_keyness" for f in files]

            # Log what was found
            logger.info("list_corpora -> analysis_type=%s, files=%s", analysis_type, files)

        return JsonResponse({"corpora": files})
    except Exception as e:
        logger.exception("Error in list_corpora")
        return JsonResponse({"error": str(e)}, status=500)
# ---------------------------------------------------------------------------

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
    # Check if this is a user text comparison upload
    comparison_mode = request.POST.get('comparison_mode', 'corpus')

    if comparison_mode == 'user_text':
        reference_file = request.FILES.get('reference_file')
        target_file = request.FILES.get('target_file')

        if not reference_file or not target_file:
            logger.warning("User text comparison attempted without both files")
            return JsonResponse({
                'error': 'Both reference_file and target_file are required',
                'success': False
            }, status=400)

        # Validate both files
        is_valid_ref, error_ref = validate_file(reference_file)
        is_valid_tgt, error_tgt = validate_file(target_file)

        if not is_valid_ref:
            logger.warning(f"Reference file validation failed: {error_ref}")
            return JsonResponse({'error': f"Reference: {error_ref}", 'success': False}, status=400)

        if not is_valid_tgt:
            logger.warning(f"Target file validation failed: {error_tgt}")
            return JsonResponse({'error': f"Target: {error_tgt}", 'success': False}, status=400)

        # Process both files
        ref_text, ref_error = process_text_file(reference_file)
        tgt_text, tgt_error = process_text_file(target_file)

        if ref_error:
            return JsonResponse({'error': f"Reference: {ref_error}", 'success': False}, status=500)

        if tgt_error:
            return JsonResponse({'error': f"Target: {tgt_error}", 'success': False}, status=500)

        logger.info(
            f"User text comparison: ref={reference_file.name} ({len(ref_text.split())} words), "
            f"target={target_file.name} ({len(tgt_text.split())} words)"
        )

        return JsonResponse({
            'success': True,
            'comparison_mode': 'user_text',
            'reference_file': {
                'filename': reference_file.name,
                'file_size': reference_file.size,
                'text_content': ref_text,
                'word_count': len(ref_text.split()),
                'char_count': len(ref_text),
            },
            'target_file': {
                'filename': target_file.name,
                'file_size': target_file.size,
                'text_content': tgt_text,
                'word_count': len(tgt_text.split()),
                'char_count': len(tgt_text),
            }
        })
    ensure_session_exists(request)
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
@require_GET
def get_corpus_preview(request):
    """
    Optional query param: ?name=<genre>
    Returns 4 preview lines. Prefer meta.preview if present.
    """
    try:
        corpus_name = request.GET.get("name")
        if corpus_name:
            meta = load_corpus_meta(corpus_name)
            # Prefer curated preview if available
            preview_lines = meta.get("preview")
            if not preview_lines:
                # Fallback: synthesize from top frequent words
                freq = meta.get("freq", {})
                words = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:20]
                preview_text = " ".join([w for w, _ in words])
                preview_lines = [preview_text[i:i+80] for i in range(0, len(preview_text), 80)][:4]
            return JsonResponse({"preview": "\n".join(preview_lines[:4])})

        # Backward compatibility (no genre provided): use legacy sample file
        try:
            with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
                corpus_text = f.read()
        except FileNotFoundError:
            return JsonResponse({"error": "No corpus files found."}, status=404)

        lines = [ln.strip() for ln in corpus_text.split("\n") if ln.strip()][:4]
        return JsonResponse({"preview": "\n".join(lines)})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# -----------------------------------------------------------------------------
# Local helper: counts-based keyness (no other files changed)
# -----------------------------------------------------------------------------
def compute_keyness_from_counts(
    uploaded_counts,
    uploaded_total,
    corpus_counts_map,
    corpus_total,
    top_n=50
):
    """
    Log-likelihood (G^2) keyness from counts.
    Returns {"results": [...], "total_significant": int}
    """
    def safe(x):
        return x if x > 0 else 1e-12

    results = []
    vocab = set(uploaded_counts.keys()) | set(corpus_counts_map.keys())

    for w in vocab:
        a = int(uploaded_counts.get(w, 0))
        b = int(corpus_counts_map.get(w, 0))
        c = max(uploaded_total - a, 0)
        d = max(corpus_total - b, 0)

        N = a + b + c + d
        if N == 0:
            continue

        row1 = a + c
        row2 = b + d
        col1 = a + b
        col2 = c + d

        E_a = safe(row1 * col1 / N)
        E_b = safe(row2 * col1 / N)
        E_c = safe(row1 * col2 / N)
        E_d = safe(row2 * col2 / N)

        G2 = 0.0
        if a > 0: G2 += 2.0 * a * math.log(a / E_a)
        if b > 0: G2 += 2.0 * b * math.log(b / E_b)
        if c > 0: G2 += 2.0 * c * math.log(c / E_c)
        if d > 0: G2 += 2.0 * d * math.log(d / E_d)

        try:
            p_value = 1.0 - chi2.cdf(G2, df=1)
        except Exception:
            p_value = 1.0

        uf = (a / uploaded_total) if uploaded_total else 0.0
        cf = (b / corpus_total) if corpus_total else 0.0
        direction = "Positive" if uf > cf else "Negative"

        results.append({
            "word": w,
            "uploaded_count": a,
            "corpus_count": b,
            "uploaded_freq": uf,
            "corpus_freq": cf,
            "g2": G2,
            "p_value": p_value,
            "keyness": direction
        })

    results.sort(key=lambda r: r["g2"], reverse=True)
    top = results[:top_n]
    total_significant = sum(1 for r in top if r["p_value"] < 0.05)

    return {"results": top, "total_significant": total_significant}


@csrf_exempt
@require_http_methods(["POST"])
def analyse_keyness(request):
    try:
        data = json.loads(request.body)

        # Get comparison mode
        comparison_mode = data.get("comparison_mode", "corpus")

        uploaded_text = data.get("uploaded_text", "")
        if not uploaded_text:
            logger.warning("Analysis request with empty uploaded_text")
            return JsonResponse({"error": "No uploaded text provided"}, status=400)

        # OPTIMIZATION 1: Limit text length to prevent memory issues
        if len(uploaded_text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text truncated from {len(uploaded_text)} to {MAX_TEXT_LENGTH} chars")
            uploaded_text = uploaded_text[:MAX_TEXT_LENGTH]

        method = data.get("method", "nltk").lower()
        filter_mode = data.get("filter_mode", "content")

        # Choose filtering function
        filter_fn = filter_all_words if filter_mode == "all" else filter_content_words

        # OPTIMIZATION 2: Process uploaded text once and clear from memory
        filtered_uploaded = filter_fn(uploaded_text)
        uploaded_total = len(filtered_uploaded)

        results_list = []
        total_significant = 0
        corpus_total = 0

        # Branch based on comparison mode
        if comparison_mode == "user_text":
            reference_text = data.get("reference_text", "")
            if not reference_text:
                logger.warning("User text comparison without reference_text")
                return JsonResponse({"error": "No reference text provided"}, status=400)

            # OPTIMIZATION 3: Limit reference text as well
            if len(reference_text) > MAX_TEXT_LENGTH:
                reference_text = reference_text[:MAX_TEXT_LENGTH]

            logger.info(
                f"User text comparison: target={len(uploaded_text.split())} words, "
                f"reference={len(reference_text.split())} words"
            )

            # Build reference counts map
            filtered_reference = filter_fn(reference_text)
            reference_counts_map = Counter([w["word"] for w in filtered_reference])
            reference_total = sum(reference_counts_map.values())

            # OPTIMIZATION 4: Clear filtered data from memory
            del filtered_reference
            gc.collect()

            # Call the selected method
            if method == "nltk":
                data_out = keyness_nltk(
                    uploaded_text,
                    corpus_counts_map=reference_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            elif method == "sklearn":
                data_out = keyness_sklearn(
                    uploaded_text,
                    corpus_counts_map=reference_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            elif method == "gensim":
                data_out = keyness_gensim(
                    uploaded_text,
                    corpus_counts_map=reference_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            elif method == "spacy":
                data_out = keyness_spacy(
                    uploaded_text,
                    corpus_counts_map=reference_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            else:
                logger.warning(f"Unknown keyness method requested: {method}")
                return JsonResponse({"error": f"Unknown method: {method}"}, status=400)

            # OPTIMIZATION 5: Clear large objects after use
            del reference_counts_map
            gc.collect()

            # Extract results
            results_list = data_out.get("results", [])
            total_significant = data_out.get("total_significant", len(results_list))
            corpus_total = reference_total

            # Ensure POS tags
            for item in results_list:
                item["pos"] = item.get("pos", item.get("pos_tag", "OTHER")).upper()

            # Save to DB
            keyness_obj = KeynessResult.objects.create(
                method=method,
                uploaded_text=uploaded_text[:10000],  # OPTIMIZATION 6: Only save first 10k chars to DB
                results=results_list,
                uploaded_total=uploaded_total,
                corpus_total=corpus_total,
            )

            logger.info(
                f"User text keyness: results={len(results_list)}, id={keyness_obj.id}"
            )

            return JsonResponse({
                "id": keyness_obj.id,
                "method": method,
                "comparison_mode": "user_text",
                "filter_mode": filter_mode,
                "results": results_list,
                "uploaded_total": uploaded_total,
                "corpus_total": corpus_total,
                "total_significant": total_significant
            })

        else:
            corpus_name = data.get("corpus_name", "").strip()
            if not corpus_name:
                return JsonResponse({"error": "No corpus_name provided"}, status=400)

            # Remove .json if already included
            corpus_name = re.sub(r'\.json$', '', corpus_name)

            # Append _keyness.json only if not already present
            if not corpus_name.endswith("_keyness"):
                filename = f"{corpus_name}_keyness.json"
            else:
                filename = f"{corpus_name}.json"

            corpus_path = KEYNESS_DIR / filename
            if not corpus_path.exists():
                return JsonResponse({"error": f"Unknown corpus_name: {filename}"}, status=400)

            # Load corpus counts from JSON
            with open(corpus_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                logger.info(f"Loaded corpus meta: {filename}")

            # Corpus counts map from JSON
            corpus_counts_map = Counter(meta.get("counts", {}))
            corpus_total = sum(corpus_counts_map.values())

            # OPTIMIZATION 7: Clear meta from memory
            del meta
            gc.collect()

            # Compute keyness using chosen method
            if method == "nltk":
                data_out = keyness_nltk(
                    uploaded_text,
                    corpus_counts_map=corpus_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            elif method == "sklearn":
                data_out = keyness_sklearn(
                    uploaded_text,
                    corpus_counts_map=corpus_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            elif method == "gensim":
                data_out = keyness_gensim(
                    uploaded_text,
                    corpus_counts_map=corpus_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            elif method == "spacy":
                data_out = keyness_spacy(
                    uploaded_text,
                    corpus_counts_map=corpus_counts_map,
                    top_n=50,
                    filter_func=filter_fn
                )
            else:
                logger.warning(f"Unknown keyness method requested: {method}")
                return JsonResponse({"error": f"Unknown method: {method}"}, status=400)

            # OPTIMIZATION 8: Clear corpus map after use
            del corpus_counts_map
            gc.collect()

            results_list = data_out.get("results", [])
            total_significant = data_out.get("total_significant", len(results_list))

            # Ensure every item has a POS tag for frontend
            for item in results_list:
                item["pos"] = item.get("pos", item.get("pos_tag", "OTHER")).upper()

            # Save to DB
            keyness_obj = KeynessResult.objects.create(
                method=method,
                uploaded_text=uploaded_text[:10000],  # Only save first 10k chars
                results=results_list,
                uploaded_total=uploaded_total,
                corpus_total=corpus_total,
            )

            logger.info(
                f"Keyness analysis completed: method={method}, "
                f"filter_mode={filter_mode}, results_count={len(results_list)}, "
                f"uploaded_total={uploaded_total}, corpus_total={corpus_total}, id={keyness_obj.id}"
            )

            # OPTIMIZATION 9: Final cleanup
            del uploaded_text, filtered_uploaded, data_out
            gc.collect()

            return JsonResponse({
                "id": keyness_obj.id,
                "method": method,
                "comparison_mode": "corpus",
                "filter_mode": filter_mode,
                "results": results_list,
                "uploaded_total": uploaded_total,
                "corpus_total": corpus_total,
                "total_significant": total_significant
            })

    except Exception as e:
        logger.exception(f"Error during keyness analysis: {e}")
        # OPTIMIZATION 10: Cleanup on error
        gc.collect()
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

@csrf_exempt
@require_http_methods(["POST"])
def get_sentences(request):
    try:
        data = json.loads(request.body)
        uploaded_text = data.get("uploaded_text", "")
        word = data.get("word", "")

        if not uploaded_text or not word:
            return JsonResponse({"error": "Missing text or word"}, status=400)

        # Extract sentences containing the word
        sentences = extract_sentences(uploaded_text, word)

        return JsonResponse({"sentences": sentences})

    except Exception as e:
        logger.exception(f"Error extracting sentences: {e}")
        return JsonResponse({"error": str(e)}, status=500)






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
        schedule_session_cleanup(request, delay_minutes=15)
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


@require_http_methods(["GET"])
def corpus_preview_keyness(request):
    # Get the genre from the query string
    genre = request.GET.get("name", "").strip()
    if not genre:
        return JsonResponse({"preview": ""})

    # Remove any trailing '.json' first
    genre = re.sub(r'\.json$', '', genre)

    # Ensure filename ends with '_keyness.json'
    if not genre.endswith("_keyness"):
        filename = f"{genre}_keyness.json"
    else:
        filename = f"{genre}.json"

    file_path = KEYNESS_DIR / filename

    if not file_path.exists():
        logger.warning(f"Corpus file not found: {file_path}")
        return JsonResponse({"error": "Corpus file not found"}, status=404)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        snippets = [item.get("snippet", "") for item in data.get("previews", [])[:5]]
        preview_text = "\n\n".join(snippets)

        logger.info(f"Generated preview text (first 200 chars): {preview_text[:200]}")
        return JsonResponse({"preview": preview_text})

    except Exception as e:
        logger.exception(f"Error generating keyness corpus preview: {e}")
        return JsonResponse({"preview": ""}, status=500)

@api_view(['POST'])
def get_keyness_summary(request):
    keyness_results = request.data.get('keyness_results', [])
    top_words = keyness_results[:50]
    prompt = (
        "You are an expert NLP analyst. Here are the top 50 key words in a text, along with their keyness scores and part-of-speech tags:\n"
        f"{top_words}\n"
        "Write a summary (2-3 paragraphs) about what these results reveal about the word choices in the text. "
        "Do not explain statistical columns; instead, interpret the meaning and possible implications of these words in the context of the document."
    )
    analysis = generate_text_with_fallback(prompt, num_predict=400, temperature=0.6)
    if not analysis:
        return Response({"error": "No response from model."}, status=500)
    return Response({"summary": analysis})

@require_http_methods(["GET"])
def corpus_meta_keyness(request):
    """
    Returns the metadata (titles, authors, genre, version) from corpus JSON files
    for displaying in the corpus information modal.
    """
    # Get the genre from the query string
    genre = request.GET.get("name", "").strip()
    if not genre:
        return JsonResponse({"error": "Genre parameter is required"}, status=400)

    # Remove any trailing '.json' first
    genre = re.sub(r'\.json$', '', genre)

    # Ensure filename ends with '_keyness.json'
    if not genre.endswith("_keyness"):
        filename = f"{genre}_keyness.json"
    else:
        filename = f"{genre}.json"

    file_path = KEYNESS_DIR / filename

    if not file_path.exists():
        logger.warning(f"Corpus metadata file not found: {file_path}")
        return JsonResponse({"error": "Corpus file not found"}, status=404)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract metadata for the modal
        metadata = {
            "genre": data.get("genre", genre),
            "version": data.get("version", "Unknown"),
            "previews": []
        }

        # Extract just title and author from each preview
        for item in data.get("previews", []):
            preview_item = {
                "title": item.get("title", "Unknown Title"),
                "author": item.get("author", "Unknown Author")
            }
            metadata["previews"].append(preview_item)

        logger.info(f"Retrieved metadata for {len(metadata['previews'])} books in {genre} corpus")
        return JsonResponse(metadata)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in corpus file {file_path}: {e}")
        return JsonResponse({"error": "Invalid JSON format"}, status=500)
    except Exception as e:
        logger.exception(f"Error reading corpus metadata: {e}")
        return JsonResponse({"error": "Internal server error"}, status=500)


@api_view(['POST'])
def get_synonyms(request):
    word = request.data.get('word', None)
    if not word:
        return Response({'error': 'No word provided.'}, status=400)

    prompt = f"""You are a linguistic expert specializing in word analysis and semantic differences.

Task: Provide exactly 5 synonyms for the word "{word}" and analyze their subtle differences.

Format your response as follows:

**Synonyms for "{word}":**

1. **[Synonym 1]**
   - Meaning: [Brief definition]
   - Difference from "{word}": [Explain the subtle difference]
   - Usage context: [When to use this instead]
   - Example: [Show both words in similar sentences to demonstrate the difference]

2. **[Synonym 2]**
   - Meaning: [Brief definition]
   - Difference from "{word}": [Explain the subtle difference]
   - Usage context: [When to use this instead]
   - Example: [Show both words in similar sentences to demonstrate the difference]

[Continue for all 5 synonyms]

**Summary:**
Write a brief paragraph explaining how choosing different synonyms can change the tone, formality, or precise meaning of your text.

Requirements:
- Choose synonyms that are genuinely interchangeable in at least some contexts
- Focus on subtle differences rather than obvious ones
- Provide concrete examples showing the difference in usage
- Consider connotation, formality level, and context appropriateness"""

    try:
        analysis = generate_text_with_fallback(prompt, num_predict=400, temperature=0.7)
        if not analysis:
            return Response({'error': 'No response from model.'}, status=500)
        return Response({
            "word": word,
            "analysis": analysis,
            "success": True
        })

    except requests.exceptions.Timeout:
        return Response({'error': 'Request timed out. The model is taking too long to respond.'}, status=504)
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        if status_code == 404:
            return Response({'error': 'Hugging Face model not found. Check HUGGINGFACE_MODEL name or repository visibility.'}, status=502)
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except requests.exceptions.RequestException as e:
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except Exception as e:
        return Response({'error': f'An error occurred: {str(e)}'}, status=500)

@api_view(['POST'])
def get_concepts(request):
    word = request.data.get('word', None)
    uploaded_text = request.data.get("uploaded_text", "")

    if not word or not uploaded_text:
        return Response({'error': 'No word or text provided.'}, status=400)

    # Reuse your sentence extraction logic
    sentences = extract_sentences(uploaded_text, word)

    # Filter like in get_sentences
    filtered_sentences = []
    word_lower = word.lower()
    for s in sentences:
        words_in_sentence = re.findall(r"\b\w[\w'-]*\b", s)
        if any(w.lower().replace("'", "") == word_lower for w in words_in_sentence):
            filtered_sentences.append(s)

    # Limit how many sentences we pass to the model (to avoid prompt bloat)
    sample_sentences = filtered_sentences[:5]
    sentences_text = "\n".join([f"- {s}" for s in sample_sentences]) if sample_sentences else "No real sentences found."

    # Build the prompt with real examples
    prompt = f"""You are a linguistic and conceptual analysis expert.

Task: Given the word "{word}", provide a breakdown of the 3–5 main concepts or senses that this word might refer to.
Use the example sentences provided below as your evidence. For each concept, select one of the provided sentences
that best illustrates this sense. Do not invent sentences; only use the given ones.

Example sentences:
{sentences_text}

Format your response as follows:

Concepts related to "{word}":

[Concept 1 Name]
Type: [Physical object, Metaphor, Abstract concept, Role, Process]
Description: [Brief definition]
Example usage: [Choose one of the given sentences, copy it exactly, and explain why it illustrates this concept]
Distinction: [How this differs from other senses]

[Continue for 3–5 concepts]

Summary:
[Why distinguishing between these senses matters for interpretation]
"""

    try:
        analysis = generate_text_with_fallback(prompt, num_predict=400, temperature=0.7)
        if not analysis:
            return Response({'error': 'No response from model.'}, status=500)

        return Response({
            "word": word,
            "analysis": analysis,
            "sentences": sample_sentences,
            "success": True
        })

    except requests.exceptions.Timeout:
        return Response({'error': 'Request timed out. The model is taking too long to respond.'}, status=504)
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        if status_code == 404:
            return Response({'error': 'Hugging Face model not found. Check HUGGINGFACE_MODEL name or repository visibility.'}, status=502)
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except requests.exceptions.RequestException as e:
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except Exception as e:
        return Response({'error': f'An error occurred: {str(e)}'}, status=500)


@api_view(['POST'])
def summarise_keyness_chart(request):
    """Generate AI summary of keyness chart using Groq."""
    log_memory_usage("summarise_keyness_chart start")

    chart_type = request.data.get('chart_type', 'bar')
    chart_title = request.data.get('title', 'Chart')
    chart_data = request.data.get('chart_data', [])

    if not chart_data:
        logger.warning("No chart data provided for summary")
        return Response({'error': 'No chart data provided.'}, status=400)

    logger.info(f"Generating summary for {chart_type} chart with {len(chart_data)} data points")

    # Limit data points to reduce prompt size
    max_data_points = 15
    chart_data = chart_data[:max_data_points]

    try:
        # Generate prompts optimized for LLM understanding
        if chart_type == "bar":
            chart_text = "\n".join([
                f"- {item['label']}: {item['value']:.3f}"
                for item in chart_data
            ])

            prompt = f"""You are an expert data analyst and computational linguist.

Task: Analyze the bar chart titled "{chart_title}" showing keyness analysis results.

Context: This chart displays the most statistically significant words from a text analysis, where higher values indicate words that are more distinctive or characteristic of the analyzed text compared to a reference corpus.

Chart Data (Top {len(chart_data)} keywords):
{chart_text}

Please provide a comprehensive analysis with the following structure:

**Summary:**
Provide a 2-3 sentence overview of the main patterns in the data.

**Key Insights:**
- Identify the top 3-5 most significant keywords and what they might indicate about the text
- Comment on the distribution pattern (steep drop-off, gradual decline, clusters)
- Note any interesting linguistic patterns (word types, themes)

**Notable Keywords:**
Highlight 3-4 specific words that stand out and briefly explain why they're significant.

Keep the analysis concise but insightful, focusing on what these keywords reveal about the text's distinctive characteristics."""

        else:  # scatter plot
            chart_text = "\n".join([
                f"- {item['label']}: Frequency={item.get('x', 0)}, Keyness={item.get('y', 0):.3f}"
                for item in chart_data
            ])

            prompt = f"""You are an expert data analyst and computational linguist.

Task: Analyze the scatter plot titled "{chart_title}" showing the relationship between word frequency and keyness scores.

Context: This visualization plots words based on their frequency (how often they appear) versus their keyness score (how distinctive they are). The most interesting words are often those with moderate-to-high frequency but very high keyness scores.

Chart Data (Top {len(chart_data)} keywords):
{chart_text}

Please provide a comprehensive analysis with the following structure:

**Summary:**
Describe the overall relationship between frequency and keyness in 2-3 sentences.

**Key Insights:**
- Identify words with high keyness but moderate frequency (these are often the most interesting)
- Comment on any outliers or unusual patterns
- Discuss the balance between common distinctive words vs. rare distinctive words

**Notable Keywords:**
Highlight 3-4 specific words that occupy interesting positions in the frequency-keyness space and explain their significance.

Focus on what the frequency-keyness relationship reveals about the text's linguistic characteristics."""

        # Clear chart_data from memory
        del chart_data
        gc.collect()

        # Generate analysis
        analysis = generate_text_with_fallback(prompt, num_predict=500, temperature=0.7)

        if not analysis:
            logger.error("No response from LLM model")
            return Response({'error': 'No response from model.'}, status=500)

        analysis = analysis.strip()

        # Clear prompt from memory
        del prompt
        gc.collect()

        logger.info(f"Summary generated successfully ({len(analysis)} chars)")
        log_memory_usage("summarise_keyness_chart end")

        return Response({
            "chart_title": chart_title,
            "chart_type": chart_type,
            "analysis": analysis,
            "success": True,
            "data_points_analyzed": min(len(request.data.get('chart_data', [])), max_data_points)
        })

    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        gc.collect()
        return Response({
            'error': 'Request timed out. The model is taking too long to respond.'
        }, status=504)

    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        logger.error(f"LLM HTTP error: {status_code} - {str(e)}")
        gc.collect()

        if status_code == 401:
            return Response({
                'error': 'Invalid API key. Please check your LLM provider API key in environment variables.'
            }, status=502)
        elif status_code == 429:
            return Response({
                'error': 'Rate limit exceeded. Please try again in a moment.'
            }, status=429)
        return Response({
            'error': f'Request to language model failed: {str(e)}'
        }, status=500)

    except ValueError as e:
        logger.error(f"LLM value error: {str(e)}")
        gc.collect()
        return Response({
            'error': str(e)
        }, status=500)

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request failed: {str(e)}")
        gc.collect()
        return Response({
            'error': f'Request to language model failed: {str(e)}'
        }, status=500)

    except Exception as e:
        logger.exception(f"Unexpected error in summarise_keyness_chart: {e}")
        gc.collect()

        # Check if it's a connection error to LLM service
        if "Connection refused" in str(e) or "Max retries exceeded" in str(e):
            return Response({
                'error': 'LLM service is not available. Please configure LLM_PROVIDER and API key in environment variables.',
                'summary_unavailable': True
            }, status=503)

        return Response({
            'error': f'An error occurred: {str(e)}'
        }, status=500)


@api_view(['POST'])
def summarise_clustering_chart(request):
    clusters = request.data.get('clusters', [])
    top_terms = request.data.get('top_terms', {})
    themes = request.data.get('themes', {})
    selected_cluster = request.data.get('selected_cluster', 'all')
    chart_title = request.data.get('title', 'Clustering Analysis')

    if not clusters:
        return Response({'error': 'No clustering data provided.'}, status=400)

    # Filter clusters if specific cluster is selected
    if selected_cluster != 'all':
        try:
            cluster_num = int(selected_cluster)
            filtered_clusters = [c for c in clusters if c.get('label') == cluster_num]
        except (ValueError, TypeError):
            filtered_clusters = clusters
    else:
        filtered_clusters = clusters

    # Prepare cluster statistics
    cluster_stats = {}
    for cluster in filtered_clusters:
        label = cluster.get('label', 'Unknown')
        if label not in cluster_stats:
            cluster_stats[label] = {
                'count': 0,
                'sample_docs': [],
                'x_coords': [],
                'y_coords': []
            }

        cluster_stats[label]['count'] += 1
        cluster_stats[label]['x_coords'].append(cluster.get('x', 0))
        cluster_stats[label]['y_coords'].append(cluster.get('y', 0))

        # Add sample document (truncated for brevity)
        doc = cluster.get('doc', '')
        if doc and len(cluster_stats[label]['sample_docs']) < 3:
            cluster_stats[label]['sample_docs'].append(doc[:100] + '...' if len(doc) > 100 else doc)

    # Generate cluster summary text
    cluster_summary = []
    total_documents = len(filtered_clusters)
    num_clusters = len(cluster_stats)

    for label, stats in sorted(cluster_stats.items()):
        # Calculate cluster position (centroid)
        avg_x = sum(stats['x_coords']) / len(stats['x_coords']) if stats['x_coords'] else 0
        avg_y = sum(stats['y_coords']) / len(stats['y_coords']) if stats['y_coords'] else 0

        # Get top terms for this cluster
        cluster_terms = top_terms.get(str(label), [])[:5]  # Top 5 terms
        terms_text = ", ".join(cluster_terms) if cluster_terms else "No terms available"

        # Get theme if available
        theme = themes.get(str(label), "No theme identified")

        cluster_info = (
            f"- Cluster {label}: {stats['count']} documents "
            f"(Position: x={avg_x:.2f}, y={avg_y:.2f})\n"
            f"  Top terms: {terms_text}\n"
            f"  Theme: {theme}"
        )

        if stats['sample_docs']:
            cluster_info += f"\n  Sample documents: {' | '.join(stats['sample_docs'][:2])}"

        cluster_summary.append(cluster_info)

    cluster_text = "\n\n".join(cluster_summary)

    # Build the analysis scope description
    scope_desc = f"all {num_clusters} clusters" if selected_cluster == 'all' else f"Cluster {selected_cluster} only"

    prompt = f"""You are an expert data scientist specializing in text clustering and unsupervised machine learning analysis.

Task: Analyze the clustering visualization titled "{chart_title}" showing document clustering results using dimensionality reduction (PCA).

Context: This scatter plot displays documents clustered using machine learning algorithms, where each point represents a document positioned in 2D space based on similarity. The x and y coordinates represent the first two principal components from PCA dimensionality reduction. Documents that are closer together are more similar in content.

Analysis Scope: {scope_desc}
Total Documents Analyzed: {total_documents}
Number of Clusters: {num_clusters}

Detailed Cluster Information:
{cluster_text}

Please provide a comprehensive analysis with the following structure:

**Executive Summary:**
Provide a 3-4 sentence overview of the clustering results, including the main patterns and overall quality of the clustering.

**Cluster Analysis:**
- Describe the spatial distribution of clusters (are they well-separated, overlapping, or forming clear groups?)
- Identify the largest and smallest clusters and what this might indicate
- Comment on any clusters that appear to be outliers or have unusual positioning
- Analyze the thematic coherence based on the top terms and identified themes

**Key Insights:**
- What do the cluster themes reveal about the underlying document collection?
- Are there any interesting relationships or patterns between clusters?
- Comment on the effectiveness of the clustering (clear separation vs. ambiguous boundaries)
- Identify any potential subclusters or hierarchical relationships

**Notable Findings:**
Highlight 3-4 specific observations about individual clusters, their content, or positioning that provide meaningful insights about the document collection.

**Recommendations:**
Suggest potential next steps for analysis or ways to improve the clustering results.

Focus on actionable insights about what these clusters reveal about the document collection's structure and content themes."""

    try:
        analysis = generate_text_with_fallback(prompt, num_predict=600, temperature=0.7)
        if not analysis:
            return Response({'error': 'No response from model.'}, status=500)

        analysis = analysis.strip()

        return Response({
            "chart_title": chart_title,
            "analysis_scope": scope_desc,
            "total_documents": total_documents,
            "num_clusters": num_clusters,
            "analysis": analysis,
            "success": True,
            "cluster_summary": cluster_stats
        })

    except requests.exceptions.Timeout:
        return Response({'error': 'Request timed out. The clustering analysis is taking too long to process.'},
                        status=504)
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        if status_code == 404:
            return Response({'error': 'Hugging Face model not found. Check HUGGINGFACE_MODEL name or repository visibility.'}, status=502)
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except requests.exceptions.RequestException as e:
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except Exception as e:
        return Response({'error': f'An error occurred: {str(e)}'}, status=500)


@api_view(['POST'])
def analyse_themes(request):
    # Accept either raw text data or clustering data
    text_documents = request.data.get('text_documents', [])
    clusters = request.data.get('clusters', [])
    top_terms = request.data.get('top_terms', {})
    themes = request.data.get('themes', {})
    analysis_title = request.data.get('title', 'Theme Analysis')

    # Determine the best data source for theme analysis
    if not text_documents and not clusters:
        return Response({'error': 'No text data or clustering data provided.'}, status=400)

    # Prepare text for analysis
    if text_documents:
        # Use original text documents if available (best for theme analysis)
        sample_size = min(50, len(text_documents))  # Limit for prompt size
        text_sample = text_documents[:sample_size]
        total_docs = len(text_documents)
        data_source = "original text documents"

        # Create text summary for prompt
        text_preview = []
        for i, doc in enumerate(text_sample[:10]):  # Show first 10 for preview
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            text_preview.append(f"Document {i + 1}: {preview}")

        documents_text = "\n\n".join(text_preview)

    else:
        # Fall back to clustering data if no original text
        sample_size = min(30, len(clusters))
        cluster_sample = clusters[:sample_size]
        total_docs = len(clusters)
        data_source = "clustered documents"

        # Create text from clustering data
        text_preview = []
        for i, cluster in enumerate(cluster_sample):
            doc = cluster.get('doc', '')
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            text_preview.append(f"Document {i + 1} (Cluster {cluster.get('label', 'N/A')}): {preview}")

        documents_text = "\n\n".join(text_preview)

    # Prepare clustering context if available
    clustering_context = ""
    if top_terms:
        cluster_info = []
        for cluster_id, terms in list(top_terms.items())[:10]:  # Top 10 clusters
            theme = themes.get(cluster_id, "No theme identified")
            cluster_info.append(f"- Cluster {cluster_id}: {', '.join(terms[:5])} (Theme: {theme})")

        if cluster_info:
            clustering_context = f"""
Additional Context from Clustering Analysis:
{chr(10).join(cluster_info)}
"""

    prompt = f"""You are an expert qualitative researcher and thematic analyst specializing in identifying patterns, themes, and topics in text data.

    Task: Conduct a comprehensive thematic analysis of the document collection titled "{analysis_title}".

    Data Source: {data_source}
    Total Documents: {total_docs}
    Sample Analyzed: {sample_size} documents

    Document Sample:
    {documents_text}
    {clustering_context}

    Instructions:
    Analyze the provided text to identify dominant themes, recurring ideas, and key topics. Look for both explicit topics (directly stated) and implicit themes (underlying patterns, sentiments, or conceptual frameworks).

    Please provide your analysis in the following structured format:

    **Major Themes (3-5 primary themes):**
    For each major theme, provide:
    - Theme name and brief description
    - Supporting evidence from the text
    - Estimated prevalence/importance
    - Key concepts and terms associated with this theme

    **Secondary Topics (3-4 supporting topics):**
    Identify important but less dominant topics that appear across the documents:
    - Topic name and description
    - How it relates to major themes
    - Frequency of occurrence

    **Conceptual Patterns:**
    - What overarching narrative or perspective emerges?
    - Are there contrasting viewpoints or tensions?
    - What underlying assumptions or frameworks are present?

    **Key Terms and Phrases:**
    List the most significant terms, phrases, or concepts that characterize this collection:
    - High-frequency meaningful terms
    - Distinctive vocabulary or jargon
    - Emotionally charged or significant phrases

    **Summary Insight:**
    Provide a 2-3 sentence synthesis of what this collection is fundamentally about - its core purpose, perspective, or focus.

    Focus on actionable insights that reveal the essential character and intellectual content of this document collection. Prioritize themes that appear across multiple documents rather than isolated topics."""

    try:
        analysis = generate_text_with_fallback(prompt, num_predict=700, temperature=0.6)
        if not analysis:
            return Response({'error': 'No response from model.'}, status=500)

        analysis = analysis.strip()

        schedule_session_cleanup(request, delay_minutes=15)
        return Response({
            "analysis_title": analysis_title,
            "data_source": data_source,
            "total_documents": total_docs,
            "documents_analyzed": sample_size,
            "analysis": analysis,
            "success": True,
            "has_clustering_context": bool(clustering_context)
        })

    except requests.exceptions.Timeout:
        return Response({'error': 'Request timed out. Theme analysis is taking too long to process.'}, status=504)
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        if status_code == 404:
            return Response({'error': 'Hugging Face model not found. Check HUGGINGFACE_MODEL name or repository visibility.'}, status=502)
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except requests.exceptions.RequestException as e:
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except Exception as e:
        return Response({'error': f'An error occurred: {str(e)}'}, status=500)


@api_view(['POST'])
def analyse_thematic_flow(request):
    text_documents = request.data.get('text_documents', [])
    clusters = request.data.get('clusters', [])
    top_terms = request.data.get('top_terms', {})
    themes = request.data.get('themes', {})
    analysis_title = request.data.get('title', 'Thematic Flow Analysis')

    if not text_documents and not clusters:
        return Response({'error': 'No text data or clustering data provided.'}, status=400)

    # Prepare text for analysis (same as analyse_themes)
    if text_documents:
        sample_size = min(50, len(text_documents))
        text_sample = text_documents[:sample_size]
        total_docs = len(text_documents)
        data_source = "original text documents"

        text_preview = []
        for i, doc in enumerate(text_sample[:10]):
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            text_preview.append(f"Document {i + 1}: {preview}")

        documents_text = "\n\n".join(text_preview)
    else:
        sample_size = min(30, len(clusters))
        cluster_sample = clusters[:sample_size]
        total_docs = len(clusters)
        data_source = "clustered documents"

        text_preview = []
        for i, cluster in enumerate(cluster_sample):
            doc = cluster.get('doc', '')
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            text_preview.append(f"Document {i + 1} (Cluster {cluster.get('label', 'N/A')}): {preview}")

        documents_text = "\n\n".join(text_preview)

    clustering_context = ""
    if top_terms:
        cluster_info = []
        for cluster_id, terms in list(top_terms.items())[:10]:
            theme = themes.get(cluster_id, "No theme identified")
            cluster_info.append(f"- Cluster {cluster_id}: {', '.join(terms[:5])} (Theme: {theme})")

        if cluster_info:
            clustering_context = f"""
Additional Context from Clustering Analysis:
{chr(10).join(cluster_info)}
"""

    prompt = f"""You are an expert in discourse analysis and thematic development, specializing in understanding how themes evolve, interconnect, and flow throughout text collections.

Task: Analyze the thematic flow and relationships in the document collection titled "{analysis_title}".

Data Source: {data_source}
Total Documents: {total_docs}
Sample Analyzed: {sample_size} documents

Document Sample:
{documents_text}
{clustering_context}

Instructions:
Examine how themes develop, connect, and flow throughout the document collection. Focus on the dynamic relationships between themes rather than static categorization.

Please provide your analysis in the following structured format:

**Thematic Progression:**
- How do themes develop across the documents?
- Are there early vs. late emerging themes?
- Do themes build upon each other or appear independently?
- Identify any narrative arc or thematic trajectory

**Theme Interconnections:**
- Which themes are closely related or complementary?
- Which themes appear together vs. separately?
- Are there hierarchical relationships (parent themes with sub-themes)?
- Map out the major theme clusters and their relationships

**Thematic Tensions and Contrasts:**
- Are there opposing or competing themes?
- Where do contradictions or tensions emerge?
- How are contrasting perspectives handled?
- Identify any dialectical relationships between themes

**Theme Transitions:**
- How does the text move between different themes?
- Are transitions smooth or abrupt?
- What linguistic or structural markers signal theme shifts?
- Identify bridging concepts that connect disparate themes

**Dominant Thematic Pathways:**
- What are the most common "routes" through the thematic landscape?
- Which theme combinations appear most frequently?
- Are there central "hub" themes that connect to many others?
- Map the primary thematic flow patterns

**Thematic Balance and Distribution:**
- Are themes evenly distributed or clustered?
- Which themes dominate the discourse?
- Are there underdeveloped themes that deserve more attention?
- Comment on the overall thematic architecture

Provide insights that reveal the dynamic, interconnected nature of themes and how they create meaning through their relationships and flow."""

    try:
        analysis = generate_text_with_fallback(prompt, num_predict=700, temperature=0.6)
        if not analysis:
            return Response({'error': 'No response from model.'}, status=500)

        analysis = analysis.strip()

        schedule_session_cleanup(request, delay_minutes=15)
        return Response({
            "analysis_title": analysis_title,
            "data_source": data_source,
            "total_documents": total_docs,
            "documents_analyzed": sample_size,
            "analysis": analysis,
            "success": True,
            "has_clustering_context": bool(clustering_context)
        })

    except requests.exceptions.Timeout:
        return Response({'error': 'Request timed out.'}, status=504)
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        if status_code == 404:
            return Response({'error': 'Hugging Face model not found. Check HUGGINGFACE_MODEL name or repository visibility.'}, status=502)
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except requests.exceptions.RequestException as e:
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except Exception as e:
        return Response({'error': f'An error occurred: {str(e)}'}, status=500)


@api_view(['POST'])
def analyse_overused_themes(request):
    text_documents = request.data.get('text_documents', [])
    clusters = request.data.get('clusters', [])
    top_terms = request.data.get('top_terms', {})
    themes = request.data.get('themes', {})
    analysis_title = request.data.get('title', 'Overused/Underused Analysis')

    if not text_documents and not clusters:
        return Response({'error': 'No text data or clustering data provided.'}, status=400)

    # Prepare text for analysis (same as above)
    if text_documents:
        sample_size = min(50, len(text_documents))
        text_sample = text_documents[:sample_size]
        total_docs = len(text_documents)
        data_source = "original text documents"

        text_preview = []
        for i, doc in enumerate(text_sample[:10]):
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            text_preview.append(f"Document {i + 1}: {preview}")

        documents_text = "\n\n".join(text_preview)
    else:
        sample_size = min(30, len(clusters))
        cluster_sample = clusters[:sample_size]
        total_docs = len(clusters)
        data_source = "clustered documents"

        text_preview = []
        for i, cluster in enumerate(cluster_sample):
            doc = cluster.get('doc', '')
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            text_preview.append(f"Document {i + 1} (Cluster {cluster.get('label', 'N/A')}): {preview}")

        documents_text = "\n\n".join(text_preview)

    clustering_context = ""
    if top_terms:
        cluster_info = []
        for cluster_id, terms in list(top_terms.items())[:10]:
            theme = themes.get(cluster_id, "No theme identified")
            cluster_info.append(f"- Cluster {cluster_id}: {', '.join(terms[:5])} (Theme: {theme})")

        if cluster_info:
            clustering_context = f"""
Additional Context from Clustering Analysis:
{chr(10).join(cluster_info)}
"""

    prompt = f"""You are an expert writing analyst and style critic specializing in identifying patterns of overuse, repetition, and stylistic habits in text collections.

Task: Analyze patterns of overuse and underuse in the document collection titled "{analysis_title}".

Data Source: {data_source}
Total Documents: {total_docs}
Sample Analyzed: {sample_size} documents

Document Sample:
{documents_text}
{clustering_context}

Instructions:
Identify elements that are overused (appearing excessively or repetitively) or underused (missing or underdeveloped) in the text collection. Focus on words, phrases, themes, and structural patterns.

Please provide your analysis in the following structured format:

**Overused Words and Phrases:**
- List specific words that appear with excessive frequency
- Identify repeated phrases or expressions (clichés, stock phrases)
- Note any buzzwords or jargon used to the point of diminishing meaning
- Provide frequency estimates and impact on readability
- Suggest alternatives or reduction strategies

**Overused Sentence Structures:**
- Identify repetitive sentence patterns (e.g., always starting with subject-verb)
- Note overreliance on specific constructions (passive voice, simple sentences, etc.)
- Comment on syntactic monotony or lack of variety
- Provide examples of the repetitive patterns
- Suggest structural variations for improvement

**Overused Themes and Concepts:**
- Identify themes that dominate to the exclusion of others
- Note concepts or ideas that are belabored or over-explained
- Highlight redundant arguments or circular reasoning
- Comment on thematic saturation points
- Suggest rebalancing or consolidation strategies

**Underused or Missing Elements:**
- Identify perspectives or angles that are neglected
- Note themes that deserve more development
- Highlight gaps in argumentation or coverage
- Suggest areas for expansion or elaboration
- Comment on missed opportunities for depth

**Stylistic Habits and Patterns:**
- Identify reflexive language choices or "verbal tics"
- Note overreliance on specific rhetorical devices
- Comment on predictable transitions or connectors
- Highlight formulaic approaches that reduce impact
- Suggest ways to break established patterns

**Impact on Overall Quality:**
- How does overuse affect readability and engagement?
- What is the cumulative effect of these patterns?
- Which overused elements most urgently need addressing?
- What would most improve the collection's variety and impact?

**Actionable Recommendations:**
Provide 3-5 concrete suggestions for reducing overuse and improving stylistic balance.

Be specific with examples and constructive in your critique. Focus on patterns that meaningfully impact the text's quality and reader experience."""

    try:
        analysis = generate_text_with_fallback(prompt, num_predict=800, temperature=0.6)
        if not analysis:
            return Response({'error': 'No response from model.'}, status=500)

        analysis = analysis.strip()

        schedule_session_cleanup(request, delay_minutes=15)
        return Response({
            "analysis_title": analysis_title,
            "data_source": data_source,
            "total_documents": total_docs,
            "documents_analyzed": sample_size,
            "analysis": analysis,
            "success": True,
            "has_clustering_context": bool(clustering_context)
        })

    except requests.exceptions.Timeout:
        return Response({'error': 'Request timed out.'}, status=504)
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", 500)
        if status_code == 404:
            return Response({'error': 'Hugging Face model not found. Check HUGGINGFACE_MODEL name or repository visibility.'}, status=502)
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except requests.exceptions.RequestException as e:
        return Response({'error': f'Request to language model failed: {str(e)}'}, status=500)
    except Exception as e:
        return Response({'error': f'An error occurred: {str(e)}'}, status=500)


def keyness_user_text(target_text, reference_text, method="nltk", top_n=50, filter_func=None):
    """
    Compute keyword statistics comparing two user texts.
    Returns:
        {
            "results": top N words as list of dicts,
            "total_significant": total unique significant words in target_text
        }
    """
    if filter_func is None:
        from .views import filter_content_words
        filter_func = filter_content_words

    # Filter / tokenize both texts
    target_tokens = filter_func(target_text)
    reference_tokens = filter_func(reference_text)

    # Count frequencies
    target_counts = Counter([t["word"] for t in target_tokens])
    reference_counts = Counter([t["word"] for t in reference_tokens])

    # Build all results with counts + POS
    all_results = []
    for word, uploaded_count in target_counts.items():
        ref_count = reference_counts.get(word, 0)

        # Skip words with 0 count in target? optional
        if uploaded_count == 0:
            continue

        # Get POS from first occurrence in target_tokens
        pos = next((t["pos"] for t in target_tokens if t["word"] == word), "OTHER")

        all_results.append({
            "word": word,
            "uploaded_count": uploaded_count,
            "sample_count": ref_count,
            "pos": pos
        })

    # Sort results by frequency in target (or implement your keyness formula)
    all_results_sorted = sorted(all_results, key=lambda x: x["uploaded_count"], reverse=True)

    # Slice top_n for frontend
    top_results = all_results_sorted[:top_n]

    return {
        "results": top_results,
        "total_significant": len(all_results_sorted)
    }

@csrf_exempt
def create_temp_corpus(request):
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]
        text = uploaded_file.read().decode("utf-8")

        # Tokenise and count frequencies
        tokens = [t.lower() for t in text.split() if len(t) > 2]
        freq = dict(Counter(tokens))
        total_tokens = len(tokens)

        out_data = {
            "genre": "temp_user_upload",
            "version": "2025-10-02",
            "previews": [
                {"title": uploaded_file.name, "author": "Unknown", "snippet": text[:500]}
            ],
            "doc_count": 1,
            "total_tokens": total_tokens,
            "counts": freq,
        }

        # Save as temp JSON in KEYNESS_DIR
        KEYNESS_DIR.mkdir(parents=True, exist_ok=True)
        temp_file = KEYNESS_DIR / "temp_user_upload_keyness.json"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        return JsonResponse({"success": True, "filename": str(temp_file)})

    return JsonResponse({"success": False, "error": "No file uploaded"}, status=400)

def health(request):
    return JsonResponse({"status": "ok"})