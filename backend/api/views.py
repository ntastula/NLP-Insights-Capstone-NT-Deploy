from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
import re
from django.conf import settings

from api.keyness.keyness_analyser import compute_keyness

CORPUS_DIR = os.path.join(settings.BASE_DIR, "api", "corpus")
SAMPLE_FILE = os.path.join(CORPUS_DIR, "sample1.txt")

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
            results = compute_keyness(uploaded_text, sample_corpus, top_n=20)
        else:
            return JsonResponse({"error": f"Unknown method: {method}"}, status=400)

        response = {
            "method": method,
            "results": results,
            "corpus_total": len(sample_corpus.split()),
            "preview": "\n".join([line for line in sample_corpus.splitlines() if line.strip()][:4])
        }

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)












# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from .keyness.keyness_analyser import compute_keyness
#
# @csrf_exempt
# def keyness_view(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             text_a = data.get("textA", "")
#             text_b = data.get("textB", "")
#             results = compute_keyness(text_a, text_b, top_n=20)
#             return JsonResponse({"results": results})
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)
#     return JsonResponse({"error": "POST request required"}, status=400)











# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# import json
# from .utils import perform_keyness_analysis, read_corpus, extract_text_from_file
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework import status
# import os
# from django.conf import settings
# import tempfile
# import docx
# import mammoth
# import numpy as np
#
# from api.keyness.keyness_analyser import KeynessAnalyser
#
# # Path to reference corpus
# CORPUS_DIR = os.path.join(settings.BASE_DIR, 'api', 'corpus')
#
# # ---------- Helper Functions ----------
# def read_corpus():
#     """Read all .txt files in the corpus directory"""
#     corpus_text = ""
#     for filename in os.listdir(CORPUS_DIR):
#         if filename.endswith(".txt"):
#             with open(os.path.join(CORPUS_DIR, filename), 'r', encoding='utf-8') as f:
#                 corpus_text += f.read() + " "
#     return corpus_text if corpus_text else None

# def safe_slice(results, n=20):
#     """Ensure results are sliceable and JSON-serializable"""
#     try:
#         if results is None:
#             return []
#         # If it's a NumPy array or pandas Series, convert to list of tuples
#         if hasattr(results, "tolist"):
#             results = results.tolist()
#         return list(results)[:n]
#     except Exception:
#         return []


# def convert_numpy(obj):
#     """Recursively convert numpy types to native Python types for JSON serialization."""
#     import numpy as np
#
#     if isinstance(obj, dict):
#         return {k: convert_numpy(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_numpy(v) for v in obj]
#     elif isinstance(obj, tuple):
#         return tuple(convert_numpy(v) for v in obj)
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     else:
#         return obj



#
#
# # ---------- API Endpoints ----------
#
# @csrf_exempt
# @require_http_methods(["GET"])
# def get_corpus_preview(request):
#     try:
#         corpus_text = read_corpus()
#         if not corpus_text:
#             return JsonResponse({"error": "No corpus files found."}, status=404)
#
#         lines = [line.strip() for line in corpus_text.split("\n") if line.strip()][:4]
#         preview = "\n".join(lines)
#         return JsonResponse({"preview": preview})
#
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)
#
#
# @csrf_exempt
# @require_http_methods(["POST"])
# def parse_document(request):
#     """Parse uploaded file (.txt, .doc, .docx)"""
#     try:
#         if 'file' not in request.FILES:
#             return JsonResponse({'error': 'No file uploaded'}, status=400)
#
#         uploaded_file = request.FILES['file']
#         file_name = uploaded_file.name.lower()
#
#         # TXT
#         if file_name.endswith('.txt'):
#             text = uploaded_file.read().decode('utf-8')
#
#         # DOCX
#         elif file_name.endswith('.docx'):
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
#                 for chunk in uploaded_file.chunks():
#                     tmp_file.write(chunk)
#                 tmp_file_path = tmp_file.name
#             try:
#                 doc = docx.Document(tmp_file_path)
#                 text = '\n'.join([p.text for p in doc.paragraphs])
#             finally:
#                 os.unlink(tmp_file_path)
#
#         # DOC
#         elif file_name.endswith('.doc'):
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp_file:
#                 for chunk in uploaded_file.chunks():
#                     tmp_file.write(chunk)
#                 tmp_file_path = tmp_file.name
#             try:
#                 with open(tmp_file_path, 'rb') as doc_file:
#                     result = mammoth.extract_raw_text(doc_file)
#                     text = result.value
#             finally:
#                 os.unlink(tmp_file_path)
#         else:
#             return JsonResponse({'error': 'Unsupported file format'}, status=400)
#
#         # Preview first 4 lines
#         lines = [line.strip() for line in text.split('\n') if line.strip()]
#         preview = '\n'.join(lines[:4])
#
#         return JsonResponse({
#             'text': text,
#             'preview': preview,
#             'word_count': len(text.split())
#         })
#
#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)
#
# @csrf_exempt
# @require_http_methods(["POST"])
# def analyse_keyness(request):
#     """Perform keyness analysis for all libraries"""
#     try:
#         data = json.loads(request.body)
#         uploaded_text = data.get("uploaded_text", "")
#         if not uploaded_text:
#             return JsonResponse({"error": "No uploaded text provided"}, status=400)
#
#         sample_corpus = read_corpus()
#         if not sample_corpus:
#             return JsonResponse({"error": "Reference corpus is empty."}, status=500)
#
#         analyser = KeynessAnalyser(uploaded_text, sample_corpus)
#
#         results = {
#             "nltk": analyser.analyse_nltk(),
#             "corpus_toolkit": analyser.analyse_corpus_toolkit(),
#             "sklearn_count": analyser.analyse_sklearn(),
#             "sklearn_tfidf": analyser.analyse_sklearn_tfidf(),
#             "gensim": analyser.analyse_gensim(),
#             "spacy": analyser.analyse_spacy(),
#         }
#
#         return JsonResponse(results)
#
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)

