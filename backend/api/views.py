from django.http import JsonResponse, HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
from django.views.decorators.http import require_GET
from rest_framework.decorators import api_view
from rest_framework.response import Response
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
    compute_keyness,
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
from django.core.files.uploadedfile import UploadedFile
from .models import KeynessResult
import logging
from pathlib import Path  # <-- ADDED (minimal import required)
import math               # <-- used by compute_keyness_from_counts


CORPUS_DIR = os.path.join(settings.BASE_DIR, "api", "corpus")
SAMPLE_FILE = os.path.join(CORPUS_DIR, "sample1.txt")
logger = logging.getLogger(__name__)

# --- Genre Corpus Meta (helpers only; no other behavior changed) -----------
# Looks for metadata-only corpora in backend/api/corpus_meta/*.json
META_DIR = Path("api/corpus_meta")
KEYNESS_DIR = Path("api/corpus_meta_keyness")

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
        files.sort(key=lambda x: (0 if x == "general_english" else 1, x))
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
            files = [f"{f}_keyness" if f != "general_english" else f"{f}_keyness" for f in files]

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
        corpus_name = data.get("corpus_name").strip()
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

        uploaded_text = data.get("uploaded_text", "")
        method = data.get("method", "nltk").lower()
        filter_mode = data.get("filter_mode", "content")

        # Choose filtering function
        filter_fn = filter_all_words if filter_mode == "all" else filter_content_words

        if not uploaded_text:
            logger.warning("Analysis request with empty uploaded_text")
            return JsonResponse({"error": "No uploaded text provided"}, status=400)

        # Tokenize/filter uploaded text
        filtered_uploaded = filter_fn(uploaded_text)
        uploaded_total = len(filtered_uploaded)

        results_list = []
        total_significant = 0
        corpus_total = 0

        # --- Load corpus counts from JSON ---
        corpus_path = KEYNESS_DIR / filename
        if not corpus_path.exists():
            return JsonResponse({"error": f"Unknown corpus_name: {filename}"}, status=400)

        with open(corpus_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            logger.info(f"Loaded corpus meta: {filename}")

        # Corpus counts map from JSON
        corpus_counts_map = Counter(meta.get("counts", {}))
        corpus_total = sum(corpus_counts_map.values())

        # --- Compute keyness using chosen method ---
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

        results_list = data_out.get("results", [])
        total_significant = data_out.get("total_significant", len(results_list))

        # Ensure every item has a POS tag for frontend
        for item in results_list:
            item["pos"] = item.get("pos", item.get("pos_tag", "OTHER")).upper()

        # Save to DB
        keyness_obj = KeynessResult.objects.create(
            method=method,
            uploaded_text=uploaded_text,
            results=results_list,
            uploaded_total=uploaded_total,
            corpus_total=corpus_total,
        )

        logger.info(
            f"Keyness analysis completed: method={method}, "
            f"filter_mode={filter_mode}, results_count={len(results_list)}, "
            f"uploaded_total={uploaded_total}, corpus_total={corpus_total}, id={keyness_obj.id}"
        )

        return JsonResponse({
            "id": keyness_obj.id,
            "method": method,
            "filter_mode": filter_mode,
            "results": results_list,
            "uploaded_total": uploaded_total,
            "corpus_total": corpus_total,
            "total_significant": total_significant
        })

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

        # Filter out sentences where the match is just "'s" or "s'"
        filtered_sentences = []
        word_lower = word.lower()
        for s in sentences:
            # Find all words in the sentence
            words_in_sentence = re.findall(r"\b\w[\w'-]*\b", s)
            # Only include sentence if the exact word (ignoring apostrophes) is present
            if any(w.lower().replace("'", "") == word_lower for w in words_in_sentence):
                filtered_sentences.append(s)

        return JsonResponse({"sentences": filtered_sentences})

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
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt
    }
    ollama_response = requests.post(ollama_url, json=payload, stream=True)
    summary_parts = []
    for line in ollama_response.iter_lines():
        if line:
            try:
                data = line.decode("utf-8")
                json_obj = json.loads(data)  # <-- fixed!
                if "response" in json_obj:
                    summary_parts.append(json_obj["response"])
            except Exception as e:
                continue
    summary_text = "".join(summary_parts)
    return Response({"summary": summary_text})

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
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(ollama_url, json=payload, timeout=60)
        response.raise_for_status()

        analysis = response.json().get("response", "")

        if not analysis:
            return Response({'error': 'No response from model.'}, status=500)

        return Response({
            "word": word,
            "analysis": analysis,
            "success": True
        })

    except requests.exceptions.Timeout as e:
        return Response({'error': 'Request timed out. The model is taking too long to respond.'}, status=504)
    except requests.exceptions.RequestException as e:
        return Response({'error': f'Request to Ollama failed: {str(e)}'}, status=500)
    except Exception as e:
        return Response({'error': f'An error occurred: {str(e)}'}, status=500)