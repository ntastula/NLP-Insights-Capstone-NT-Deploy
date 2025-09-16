from django.urls import path
from . import views

urlpatterns = [
    path("corpora/", views.list_corpora, name="list_corpora"),                   # NEW
    path("corpus-preview/", views.get_corpus_preview, name="corpus_preview"),    # UPDATED (accepts ?name=)

    path("analyse-keyness/", views.analyse_keyness, name="keyness_view"),
    path('upload-files/', views.upload_files, name='upload_files'),
    path("analyse-sentiment/", views.analyse_sentiment, name="analyse-sentiment"),
    path("get-sentences/", views.get_sentences, name="get_sentences"),

]





