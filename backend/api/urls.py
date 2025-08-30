from django.urls import path
from . import views
from .views import upload_files

urlpatterns = [
    path("analyse-keyness/", views.analyse_keyness, name="keyness_view"),
    path('corpus-preview/', views.get_corpus_preview, name='corpus-preview'),
    path('upload-files/', views.upload_files, name='upload-files'),
    # path('get-user-files/', views.get_user_files, name='get_user_files'),
    # path('delete-file/', views.delete_user_file, name='delete_user_file'),
    # path('clear-data/', views.clear_user_data, name='clear_user_data'),
    # path('save-analysis/', views.save_analysis, name='save_analysis'),
]

