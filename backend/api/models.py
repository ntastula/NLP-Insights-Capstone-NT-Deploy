from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid


class UserSession(models.Model):
    """Track user sessions and their data"""
    session_key = models.CharField(max_length=40, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)  # For logged-in users
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'user_sessions'

    def __str__(self):
        return f"Session {self.session_key[:8]}... ({'User' if self.user else 'Anonymous'})"


class UserUploadedFile(models.Model):
    """Store user uploaded files with session isolation"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(UserSession, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    file_size = models.IntegerField()
    file_type = models.CharField(max_length=50)
    text_content = models.TextField()
    word_count = models.IntegerField(default=0)
    char_count = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'user_uploaded_files'
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.original_filename} ({self.session.session_key[:8]}...)"


class UserAnalysisHistory(models.Model):
    """Store analysis results per session"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(UserSession, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=50)  # 'keyness', 'sentiment', etc.
    input_text_preview = models.TextField(max_length=500)  # First 500 chars
    results = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'user_analysis_history'
        ordering = ['-created_at']