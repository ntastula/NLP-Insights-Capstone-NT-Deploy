import os
import re
import magic
import hashlib
from django.conf import settings
from .exceptions import FileValidationError


class FileValidator:
    """Handles all file validation logic"""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_FILE_SIZE = 100  # 100 bytes
    ALLOWED_EXTENSIONS = {'.txt', '.docx', '.pdf', '.rtf'}
    ALLOWED_MIME_TYPES = {
        'text/plain',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/pdf',
        'application/rtf',
        'text/rtf'
    }

    @staticmethod
    def validate_file_size(file_size):
        """Validate file size"""
        if file_size > FileValidator.MAX_FILE_SIZE:
            raise FileValidationError(
                f'File too large. Maximum size: {FileValidator.MAX_FILE_SIZE // (1024 * 1024)}MB'
            )
        if file_size < FileValidator.MIN_FILE_SIZE:
            raise FileValidationError(
                f'File too small. Minimum size: {FileValidator.MIN_FILE_SIZE} bytes'
            )

    @staticmethod
    def validate_extension(filename):
        """Validate file extension"""
        if not filename:
            raise FileValidationError("Filename cannot be empty")

        ext = os.path.splitext(filename.lower())[1]
        if ext not in FileValidator.ALLOWED_EXTENSIONS:
            raise FileValidationError(
                f"File extension '{ext}' not allowed. Allowed: {', '.join(FileValidator.ALLOWED_EXTENSIONS)}"
            )

    @staticmethod
    def validate_mime_type(file_path):
        """Validate MIME type"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            if mime_type not in FileValidator.ALLOWED_MIME_TYPES:
                raise FileValidationError(f"File type '{mime_type}' not allowed")
            return mime_type
        except Exception as e:
            raise FileValidationError(f"Could not determine file type: {str(e)}")


class TextValidator:
    """Handles text content validation"""

    MIN_TEXT_LENGTH = 50
    MAX_TEXT_LENGTH = 1000000  # 1MB

    @staticmethod
    def validate_text_content(text):
        """Validate extracted text content"""
        if not text or len(text.strip()) < TextValidator.MIN_TEXT_LENGTH:
            raise FileValidationError(
                f"Text must contain at least {TextValidator.MIN_TEXT_LENGTH} characters"
            )

        if len(text) > TextValidator.MAX_TEXT_LENGTH:
            raise FileValidationError("Text is too long (max 1MB)")

        # Check for malicious content
        script_pattern = re.compile(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', re.IGNORECASE)
        if script_pattern.search(text):
            raise FileValidationError("Text contains potentially malicious content")
