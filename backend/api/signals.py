import os, shutil, tempfile
from django.contrib.sessions.models import Session
from django.db.models.signals import post_delete
from django.dispatch import receiver

@receiver(post_delete, sender=Session)
def cleanup_session_files(sender, instance, **kwargs):
    session_dir = os.path.join(tempfile.gettempdir(), f"uploads_{instance.session_key}")
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)