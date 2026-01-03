"""
Document upload handling.
Manages file uploads with validation and storage.
"""

import logging
from pathlib import Path
from typing import List, Optional
import streamlit as st
from config import settings

logger = logging.getLogger(__name__)


class DocumentUploader:
    """Handles document upload operations."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize uploader.

        Args:
            data_dir: Directory for storing uploaded documents
        """
        self.data_dir = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_extensions = settings.get_allowed_extensions()
        self.max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def validate_file(self, file) -> tuple[bool, Optional[str]]:
        """
        Validate uploaded file.

        Args:
            file: Streamlit UploadedFile object

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension
        file_ext = Path(file.name).suffix.lower().lstrip('.')
        if file_ext not in self.allowed_extensions:
            return False, (
                f"File type '.{file_ext}' not allowed. "
                f"Allowed types: {', '.join(self.allowed_extensions)}"
            )

        # Check file size
        if file.size > self.max_size_bytes:
            max_mb = self.max_size_bytes / 1024 / 1024
            actual_mb = file.size / 1024 / 1024
            return False, (
                f"File too large ({actual_mb:.1f}MB). "
                f"Maximum size: {max_mb:.0f}MB"
            )

        # Check if file is empty
        if file.size == 0:
            return False, "File is empty"

        return True, None

    def save_file(self, file) -> tuple[bool, Optional[str], Optional[Path]]:
        """
        Save uploaded file to data directory.

        Args:
            file: Streamlit UploadedFile object

        Returns:
            Tuple of (success, error_message, file_path)
        """
        try:
            # Validate file
            is_valid, error = self.validate_file(file)
            if not is_valid:
                return False, error, None

            # Sanitize filename
            safe_filename = self._sanitize_filename(file.name)
            file_path = self.data_dir / safe_filename

            # Check if file already exists
            if file_path.exists():
                return False, f"File '{safe_filename}' already exists", None

            # Save file
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            logger.info(f"File saved: {file_path}")
            return True, None, file_path

        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return False, f"Failed to save file: {str(e)}", None

    def delete_file(self, filename: str) -> tuple[bool, Optional[str]]:
        """
        Delete a document from data directory.

        Args:
            filename: Name of file to delete

        Returns:
            Tuple of (success, error_message)
        """
        try:
            file_path = self.data_dir / filename

            if not file_path.exists():
                return False, f"File '{filename}' not found"

            file_path.unlink()
            logger.info(f"File deleted: {file_path}")
            return True, None

        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False, f"Failed to delete file: {str(e)}"

    def list_documents(self) -> List[dict]:
        """
        List all documents in data directory.

        Returns:
            List of document info dicts
        """
        documents = []

        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                documents.append({
                    "name": file_path.name,
                    "path": file_path,
                    "size_mb": file_path.stat().st_size / 1024 / 1024,
                    "extension": file_path.suffix.lower().lstrip('.'),
                    "modified": file_path.stat().st_mtime
                })

        # Sort by modified time (newest first)
        documents.sort(key=lambda x: x["modified"], reverse=True)

        return documents

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other issues.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename
        """
        # Remove path separators
        filename = filename.replace('/', '_').replace('\\', '_')

        # Remove dangerous characters
        dangerous_chars = ['..', '<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')

        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1)
            filename = name[:250] + '.' + ext

        return filename