"""Simple uploader utility to validate and move files into the data directory."""

import logging
from pathlib import Path
import shutil
from typing import Tuple
from config import settings

logger = logging.getLogger(__name__)


class Uploader:
    """Validates and saves uploaded files to the DATA_DIR."""

    def __init__(self, data_dir: Path | str = None):
        self.data_dir = Path(data_dir) if data_dir else settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, file_path: Path) -> Tuple[bool, str]:
        """Validate file size and extension.

        Returns tuple of (ok, message).
        """
        if not file_path.exists():
            return False, "File not found"

        # Check size
        size_mb = file_path.stat().st_size / 1024 / 1024
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            return False, f"File too large ({size_mb:.1f} MB)"

        # Check extension
        allowed = settings.get_allowed_extensions()
        if file_path.suffix.lower().lstrip('.') not in allowed:
            return False, f"Extension not allowed. Allowed: {', '.join(allowed)}"

        return True, "OK"

    def save(self, file_path: Path) -> Path:
        """Save file into DATA_DIR and return destination Path."""
        ok, msg = self.validate(file_path)
        if not ok:
            raise ValueError(msg)

        dest = self.data_dir / file_path.name
        shutil.copy2(file_path, dest)
        logger.info(f"Saved uploaded file to {dest}")
        return dest
