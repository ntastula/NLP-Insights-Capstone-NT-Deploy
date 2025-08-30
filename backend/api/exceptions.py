class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass

class RateLimitExceeded(Exception):
    """Custom exception for rate limiting"""
    pass

class ContentProcessingError(Exception):
    """Custom exception for content processing errors"""
    pass