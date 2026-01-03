"""Simple validation helpers used across the project."""

from pathlib import Path
from config import settings


def is_allowed_extension(file_name: str) -> bool:
    ext = Path(file_name).suffix.lower().lstrip('.')
    return ext in settings.get_allowed_extensions()


def is_within_size_limit(file_path: Path) -> bool:
    size_mb = file_path.stat().st_size / 1024 / 1024
    return size_mb <= settings.MAX_UPLOAD_SIZE_MB
