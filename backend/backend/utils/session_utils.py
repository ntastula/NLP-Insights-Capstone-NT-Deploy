from django.contrib.sessions.models import Session
from django.contrib.sessions.backends.db import SessionStore
from django.utils import timezone
from datetime import timedelta
from api.models import UserSession, UserUploadedFile, UserAnalysisHistory
import os
import tempfile


class SessionManager:
    """Manage user sessions and data isolation"""

    @staticmethod
    def get_or_create_user_session(request):
        """Get or create a UserSession for the current request"""
        session_key = request.session.session_key

        # Create Django session if it doesn't exist
        if not session_key:
            request.session.create()
            session_key = request.session.session_key

        # Get or create UserSession
        user_session, created = UserSession.objects.get_or_create(
            session_key=session_key,
            defaults={
                'user': request.user if request.user.is_authenticated else None,
                'is_active': True
            }
        )

        # Update last activity
        if not created:
            user_session.last_activity = timezone.now()
            user_session.save(update_fields=['last_activity'])

        return user_session

    @staticmethod
    def get_user_files(request):
        """Get all files for the current user's session"""
        user_session = SessionManager.get_or_create_user_session(request)
        return UserUploadedFile.objects.filter(session=user_session)

    @staticmethod
    def save_user_file(request, filename, original_filename, file_size, file_type, text_content):
        """Save a file for the current user's session"""
        user_session = SessionManager.get_or_create_user_session(request)

        user_file = UserUploadedFile.objects.create(
            session=user_session,
            filename=filename,
            original_filename=original_filename,
            file_size=file_size,
            file_type=file_type,
            text_content=text_content,
            word_count=len(text_content.split()),
            char_count=len(text_content)
        )

        return user_file

    @staticmethod
    def delete_user_file(request, file_id):
        """Delete a specific file for the current user"""
        user_session = SessionManager.get_or_create_user_session(request)
        try:
            user_file = UserUploadedFile.objects.get(
                id=file_id,
                session=user_session
            )
            user_file.delete()
            return True
        except UserUploadedFile.DoesNotExist:
            return False

    @staticmethod
    def clear_user_data(request):
        """Clear all data for the current user's session"""
        user_session = SessionManager.get_or_create_user_session(request)

        # Delete files and analysis history
        UserUploadedFile.objects.filter(session=user_session).delete()
        UserAnalysisHistory.objects.filter(session=user_session).delete()

        return True

    @staticmethod
    def save_analysis_result(request, analysis_type, input_text_preview, results):
        """Save analysis results for the current session"""
        user_session = SessionManager.get_or_create_user_session(request)

        analysis = UserAnalysisHistory.objects.create(
            session=user_session,
            analysis_type=analysis_type,
            input_text_preview=input_text_preview[:500],  # Limit preview
            results=results
        )

        return analysis

    @staticmethod
    def get_user_analysis_history(request, analysis_type=None):
        """Get analysis history for the current session"""
        user_session = SessionManager.get_or_create_user_session(request)

        queryset = UserAnalysisHistory.objects.filter(session=user_session)
        if analysis_type:
            queryset = queryset.filter(analysis_type=analysis_type)

        return queryset

    @staticmethod
    def cleanup_old_sessions(days_old=7):
        """Clean up old sessions and their data"""
        cutoff_date = timezone.now() - timedelta(days=days_old)

        # Get old sessions
        old_sessions = UserSession.objects.filter(last_activity__lt=cutoff_date)

        # Delete associated data
        for session in old_sessions:
            UserUploadedFile.objects.filter(session=session).delete()
            UserAnalysisHistory.objects.filter(session=session).delete()

        # Delete the sessions
        deleted_count = old_sessions.count()
        old_sessions.delete()

        return deleted_count

def list_session_temp_folders(delete_stale=False):
        """
        List all session-based temp folders and optionally delete stale ones.
        """
        temp_root = tempfile.gettempdir()
        prefix = "uploads_"

        # active session keys in DB
        active_sessions = set(Session.objects.values_list("session_key", flat=True))

        found_folders = []

        for name in os.listdir(temp_root):
            if not name.startswith(prefix):
                continue

            folder_path = os.path.join(temp_root, name)
            session_id = name[len(prefix):]

            status = "active" if session_id in active_sessions else "stale"
            found_folders.append((folder_path, status))

            if status == "stale" and delete_stale:
                try:
                    for f in os.listdir(folder_path):
                        os.remove(os.path.join(folder_path, f))
                    os.rmdir(folder_path)
                    print(f"✅ Deleted stale folder: {folder_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete {folder_path}: {e}")

        return found_folders

# --- US7 helpers -------------------------------------------------------------
import os, tempfile, threading, shutil

def ensure_session_exists(request):
    """
    Guarantee a Django session key so we can isolate temp files per user.
    """
    if not hasattr(request, "session"):
        return
    if not request.session.session_key:
        # Touch the session to force creation
        request.session.save()

def _session_temp_dir(session_key: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"uploads_{session_key}")

def schedule_session_cleanup(request, delay_minutes: int = 15):
    """
    Schedule deletion of this session's temp folder after analysis completes.
    This complements your existing post_delete(Session) signal cleanup.
    """
    if not hasattr(request, "session"):
        return
    session_key = request.session.session_key
    if not session_key:
        return
    folder = _session_temp_dir(session_key)

    def _cleanup():
        try:
            if os.path.exists(folder):
                shutil.rmtree(folder, ignore_errors=True)
        except Exception:
            pass

    t = threading.Timer(delay_minutes * 60, _cleanup)
    t.daemon = True
    t.start()