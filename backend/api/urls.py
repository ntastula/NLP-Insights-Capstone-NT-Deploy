from django.urls import path
from . import views
from .clustering.clustering_analyser import clustering_analysis

urlpatterns = [
    path("corpora/", views.list_corpora, name="list_corpora"),                   # NEW
    path("corpus-preview/", views.get_corpus_preview, name="corpus_preview"),    # UPDATED (accepts ?name=)

    path("analyse-keyness/", views.analyse_keyness, name="keyness_view"),
    path('upload-files/', views.upload_files, name='upload_files'),
    path("analyse-sentiment/", views.analyse_sentiment, name="analyse-sentiment"),
    path("get-sentences/", views.get_sentences, name="get_sentences"),
    path('clustering-analysis/', clustering_analysis, name='clustering-analysis'),
    path("corpus-preview-keyness/", views.corpus_preview_keyness, name="corpus_preview_keyness"),
    path('get-keyness-summary/', views.get_keyness_summary, name="get_keyness_summary"),
    path('corpus-meta-keyness/', views.corpus_meta_keyness, name='corpus_meta_keyness'),
    path('get-synonyms/', views.get_synonyms, name='get_synonyms'),
]





