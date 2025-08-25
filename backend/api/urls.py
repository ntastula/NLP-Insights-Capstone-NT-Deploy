from django.urls import path
from . import views

urlpatterns = [
    # Corpus preview
    path('corpus-preview/', views.get_corpus_preview, name='corpus-preview'),

    # Parse uploaded document
    path('parse-document/', views.parse_document, name='parse-document'),

    # Keyness analysis
    path('analyse-keyness/', views.analyse_keyness, name='analyse-keyness'),
]
