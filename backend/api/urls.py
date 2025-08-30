from django.urls import path
from . import views
from .views import upload_files

urlpatterns = [
    path("analyse-keyness/", views.analyse_keyness, name="keyness_view"),
    path('corpus-preview/', views.get_corpus_preview, name='corpus-preview'),
    path('upload-files/', upload_files, name='upload-files'),
]

