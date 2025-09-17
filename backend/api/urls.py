from django.urls import path
from . import views
from .clustering.clustering_analyser import clustering_analysis

urlpatterns = [
    path("analyse-keyness/", views.analyse_keyness, name="keyness_view"),
    path('corpus-preview/', views.get_corpus_preview, name='corpus-preview'),
    path('upload-files/', views.upload_files, name='upload_files'),
    path("analyse-sentiment/", views.analyse_sentiment, name="analyse-sentiment"),
    path("get-sentences/", views.get_sentences, name="get_sentences"),
    path('clustering-analysis/', clustering_analysis, name='clustering-analysis'),

]





