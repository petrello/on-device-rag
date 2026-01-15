"""
Document upload handling.

Manages file uploads with validation and storage to the data directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from config import settings

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)


class DocumentUploader:
    """
    Handles document upload operations.

    Validates, sanitizes, and stores uploaded files in the data directory.

    Attributes:
        data_dir: Directory for storing documents.
        allowed_extensions: List of allowed file extensions.
        max_size_bytes: Maximum file size in bytes.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize the uploader.

        Args:
            data_dir: Directory for storing uploaded documents.
        """
        self.data_dir: Path = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_extensions: List[str] = settings.get_allowed_extensions()
        self.max_size_bytes: int = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def validate_file(self, file: UploadedFile) -> Tuple[bool, Optional[str]]:
        """
        Validate an uploaded file.

        Args:
            file: Streamlit UploadedFile object.

        Returns:
            Tuple of (is_valid, error_message).
        """
        file_ext = Path(file.name).suffix.lower().lstrip('.')

        if file_ext not in self.allowed_extensions:
            return False, (
                f"File type '.{file_ext}' not allowed. "
                f"Allowed: {', '.join(self.allowed_extensions)}"
            )

        if file.size > self.max_size_bytes:
            max_mb = self.max_size_bytes / 1024 / 1024
            actual_mb = file.size / 1024 / 1024
            return False, f"File too large ({actual_mb:.1f}MB). Max: {max_mb:.0f}MB"

        if file.size == 0:
            return False, "File is empty"

        return True, None

    def save_file(
        self,
        file: UploadedFile,
    ) -> Tuple[bool, Optional[str], Optional[Path]]:
        """
        Save an uploaded file to the data directory.

        Args:
            file: Streamlit UploadedFile object.

        Returns:
            Tuple of (success, error_message, file_path).
        """
        try:
            is_valid, error = self.validate_file(file)
            if not is_valid:
                return False, error, None

            safe_filename = self._sanitize_filename(file.name)
            file_path = self.data_dir / safe_filename

            if file_path.exists():
                return False, f"File '{safe_filename}' already exists", None

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            logger.info(f"File saved: {file_path}")
            return True, None, file_path

        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return False, f"Failed to save file: {e}", None

    def delete_file(self, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Delete a document from the data directory.

        Args:
            filename: Name of file to delete.

        Returns:
            Tuple of (success, error_message).
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
            return False, f"Failed to delete file: {e}"

    def list_documents(self) -> List[Dict]:
        """
        List all documents in the data directory.

        Returns:
            List of document info dictionaries, sorted by modification time.
        """
        documents: List[Dict] = []

        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith('.'):
                documents.append({
                    "name": file_path.name,
                    "path": file_path,
                    "size_mb": file_path.stat().st_size / 1024 / 1024,
                    "extension": file_path.suffix.lower().lstrip('.'),
                    "modified": file_path.stat().st_mtime,
                })

        documents.sort(key=lambda x: x["modified"], reverse=True)
        return documents

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal attacks.

        Args:
            filename: Original filename.

        Returns:
            Sanitized filename safe for filesystem use.
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