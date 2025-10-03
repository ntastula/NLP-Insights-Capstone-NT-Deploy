from django.urls import path
from . import views
from .clustering.clustering_analyser import clustering_analysis

urlpatterns = [
    path("corpora/", views.list_corpora, name="list_corpora"),
    path("corpus-preview/", views.get_corpus_preview, name="corpus_preview"),
    path("analyse-keyness/", views.analyse_keyness, name="keyness_view"),
    path('upload-files/', views.upload_files, name='upload_files'),
    path("analyse-sentiment/", views.analyse_sentiment, name="analyse-sentiment"),
    path("get-sentences/", views.get_sentences, name="get_sentences"),
    path('clustering-analysis/', clustering_analysis, name='clustering-analysis'),
    path("corpus-preview-keyness/", views.corpus_preview_keyness, name="corpus_preview_keyness"),
    path('get-keyness-summary/', views.get_keyness_summary, name="get_keyness_summary"),
    path('corpus-meta-keyness/', views.corpus_meta_keyness, name='corpus_meta_keyness'),
    path('get-synonyms/', views.get_synonyms, name='get_synonyms'),
    path('get-concepts/', views.get_concepts, name='get_concepts'),
    path('summarise-keyness-chart/', views.summarise_keyness_chart, name='summarise_keyness_chart'),
    path('summarise-clustering-chart/', views.summarise_clustering_chart, name='summarise_clustering_chart'),
    path('analyse-themes/', views.analyse_themes, name='analyse_themes'),
    path('analyse-thematic-flow/', views.analyse_thematic_flow, name='analyse_thematic_flow'),
    path('analyse-overused-themes/', views.analyse_overused_themes, name='analyse_overused_themes'),
    path('create-temp-corpus/', views.create_temp_corpus, name='create_temp_corpus'),

]





