from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import os
import re
from django.conf import settings
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from api.keyness.keyness_analyser import compute_keyness

CORPUS_DIR = os.path.join(settings.BASE_DIR, "api", "corpus")
SAMPLE_FILE = os.path.join(CORPUS_DIR, "sample1.txt")

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
