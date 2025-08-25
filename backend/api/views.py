from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .utils import perform_keyness_analysis, read_corpus, extract_text_from_file


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
def parse_document(request):
    try:
        if "file" not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        uploaded_file = request.FILES["file"]
        result = extract_text_from_file(uploaded_file)
        return JsonResponse(result)

    except ValueError as ve:
        return JsonResponse({"error": str(ve)}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def analyse_keyness(request):
    try:
        data = json.loads(request.body)
        uploaded_text = data.get("uploaded_text", "")
        if not uploaded_text:
            return JsonResponse({"error": "No uploaded text provided"}, status=400)

        sample_corpus = read_corpus()
        if not sample_corpus:
            return JsonResponse({"error": "Reference corpus is empty."}, status=500)

        results = perform_keyness_analysis(uploaded_text, sample_corpus)
        return JsonResponse(results)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)