"""Document processing utilities.

Provides a simple DocumentProcessor wrapper around LlamaIndex SimpleDirectoryReader
and basic metadata extraction for local files.
"""

import logging
from pathlib import Path
from typing import List
from llama_index.core.schema import Document
from config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Simple document processor to load and normalize files from DATA_DIR."""

    def __init__(self, data_dir: Path | str = None):
        self.data_dir = Path(data_dir) if data_dir else settings.DATA_DIR

    def list_documents(self) -> List[Path]:
        """List document file paths in the data directory."""
        files = [p for p in self.data_dir.glob("**/*") if p.is_file()]
        logger.debug(f"Found {len(files)} files in {self.data_dir}")
        return files

    def to_documents(self) -> List[Document]:
        """Convert files to LlamaIndex Documents with minimal metadata.

        This implementation only supports simple text extraction for .txt and .md files.
        For PDFs/DOCX the code will store filename metadata and let downstream
        readers handle detailed parsing.
        """
        docs: List[Document] = []

        for f in self.list_documents():
            metadata = {"file_name": f.name}

            try:
                if f.suffix.lower() in [".txt", ".md"]:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                    docs.append(Document(text=text, metadata=metadata))
                else:
                    # For binary formats we create a placeholder document
                    docs.append(Document(text="", metadata=metadata))
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")

        logger.info(f"Converted {len(docs)} files to Document objects")
        return docs
