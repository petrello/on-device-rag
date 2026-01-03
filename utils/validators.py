"""
Input validation utilities.
Validates user inputs and system configurations.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def validate_query(query: str, min_length: int = 1, max_length: int = 5000) -> Tuple[bool, Optional[str]]:
    """
    Validate user query.

    Args:
        query: User query string
        min_length: Minimum query length
        max_length: Maximum query length

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if empty
    if not query or not query.strip():
        return False, "Query cannot be empty"

    # Check length
    if len(query) < min_length:
        return False, f"Query too short (minimum {min_length} characters)"

    if len(query) > max_length:
        return False, f"Query too long (maximum {max_length} characters)"

    # Check for suspicious patterns (basic security)
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected in query: {pattern}")
            return False, "Query contains invalid content"

    return True, None


def validate_file_path(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate file path for security.

    Args:
        file_path: Path to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Convert to Path object
        path = Path(file_path).resolve()

        # Check if file exists
        if not path.exists():
            return False, f"File does not exist: {path}"

        # Check if it's a file (not directory)
        if not path.is_file():
            return False, f"Path is not a file: {path}"

        # Check for path traversal attempts
        if '..' in str(path):
            return False, "Invalid path (path traversal detected)"

        return True, None

    except Exception as e:
        return False, f"Invalid file path: {str(e)}"


def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate filename.

    Args:
        filename: Filename to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if empty
    if not filename or not filename.strip():
        return False, "Filename cannot be empty"

    # Check length
    if len(filename) > 255:
        return False, "Filename too long (max 255 characters)"

    # Check for invalid characters
    invalid_chars = ['/', '\\', '<', '>', ':', '"', '|', '?', '*', '\0']
    for char in invalid_chars:
        if char in filename:
            return False, f"Filename contains invalid character: '{char}'"

    # Check for reserved names (Windows)
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'LPT1', 'LPT2']
    name_without_ext = filename.rsplit('.', 1)[0].upper()
    if name_without_ext in reserved_names:
        return False, f"Filename uses reserved name: {name_without_ext}"

    return True, None


def validate_extension(filename: str, allowed_extensions: list) -> Tuple[bool, Optional[str]]:
    """
    Validate file extension.

    Args:
        filename: Filename to check
        allowed_extensions: List of allowed extensions (without dot)

    Returns:
        Tuple of (is_valid, error_message)
    """
    extension = Path(filename).suffix.lower().lstrip('.')

    if not extension:
        return False, "File has no extension"

    if extension not in [ext.lower() for ext in allowed_extensions]:
        return False, (
            f"File type '.{extension}' not allowed. "
            f"Allowed types: {', '.join(allowed_extensions)}"
        )

    return True, None


def validate_file_size(size_bytes: int, max_size_mb: int) -> Tuple[bool, Optional[str]]:
    """
    Validate file size.

    Args:
        size_bytes: File size in bytes
        max_size_mb: Maximum size in MB

    Returns:
        Tuple of (is_valid, error_message)
    """
    max_bytes = max_size_mb * 1024 * 1024

    if size_bytes > max_bytes:
        actual_mb = size_bytes / 1024 / 1024
        return False, (
            f"File too large ({actual_mb:.2f}MB). "
            f"Maximum size: {max_size_mb}MB"
        )

    if size_bytes == 0:
        return False, "File is empty"

    return True, None


def validate_chunk_size(chunk_size: int, min_size: int = 50, max_size: int = 5000) -> Tuple[bool, Optional[str]]:
    """
    Validate chunk size parameter.

    Args:
        chunk_size: Chunk size to validate
        min_size: Minimum allowed size
        max_size: Maximum allowed size

    Returns:
        Tuple of (is_valid, error_message)
    """
    if chunk_size < min_size:
        return False, f"Chunk size too small (minimum {min_size})"

    if chunk_size > max_size:
        return False, f"Chunk size too large (maximum {max_size})"

    return True, None


def validate_top_k(top_k: int, min_k: int = 1, max_k: int = 50) -> Tuple[bool, Optional[str]]:
    """
    Validate top_k parameter.

    Args:
        top_k: Top-K value to validate
        min_k: Minimum allowed value
        max_k: Maximum allowed value

    Returns:
        Tuple of (is_valid, error_message)
    """
    if top_k < min_k:
        return False, f"Top-K too small (minimum {min_k})"

    if top_k > max_k:
        return False, f"Top-K too large (maximum {max_k})"

    return True, None


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input.

    Args:
        text: Text to sanitize
        max_length: Maximum length (truncate if longer)

    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace('\x00', '')

    # Normalize whitespace
    text = ' '.join(text.split())

    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length] + '...'

    return text


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """
    Check if target path is safely within base directory.

    Args:
        base_dir: Base directory
        target_path: Target path to check

    Returns:
        True if safe, False otherwise
    """
    try:
        base_dir = base_dir.resolve()
        target_path = target_path.resolve()

        # Check if target is under base
        return target_path.is_relative_to(base_dir)

    except Exception:
        return False