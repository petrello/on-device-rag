"""
Document processing pipeline.

Handles loading, validation, and preprocessing of documents before indexing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from llama_index.core import Document, SimpleDirectoryReader

from config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents for RAG indexing.

    Handles loading from disk, text cleaning, validation, and statistics
    collection for various document formats.

    Attributes:
        data_dir: Directory containing source documents.
        allowed_extensions: List of valid file extensions.
    """

    __slots__ = ('data_dir', 'allowed_extensions')

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize the document processor.

        Args:
            data_dir: Path to documents directory. Defaults to settings.DATA_DIR.
        """
        self.data_dir: Path = data_dir or settings.DATA_DIR
        self.allowed_extensions: List[str] = settings.get_allowed_extensions()

    def load_documents(
        self,
        file_paths: Optional[List[Path]] = None,
    ) -> List[Document]:
        """
        Load documents from disk.

        Args:
            file_paths: Specific files to load. If None, loads all from data_dir.

        Returns:
            List of loaded Document objects.

        Raises:
            Exception: If document loading fails.
        """
        try:
            if file_paths:
                documents = []
                for file_path in file_paths:
                    if file_path.exists():
                        reader = SimpleDirectoryReader(input_files=[str(file_path)])
                        docs = reader.load_data()
                        documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} doc(s) from {file_path.name}")
            else:
                if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
                    logger.warning("No documents found in data directory")
                    return []

                reader = SimpleDirectoryReader(
                    str(self.data_dir),
                    recursive=False,
                    filename_as_id=True,
                )
                documents = reader.load_data()
                logger.info(f"Loaded {len(documents)} documents from {self.data_dir}")

            return documents

        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean and preprocess documents before indexing.

        Args:
            documents: Raw documents to preprocess.

        Returns:
            List of preprocessed documents.
        """
        processed: List[Document] = []

        for doc in documents:
            try:
                text = self._clean_text(doc.text)

                if len(text.strip()) < 10:
                    logger.warning("Skipping document with insufficient text")
                    continue

                processed_doc = Document(
                    text=text,
                    metadata={**doc.metadata, "processed": True},
                )
                processed.append(processed_doc)

            except Exception as e:
                logger.error(f"Failed to preprocess document: {e}")
                continue

        logger.info(f"Preprocessed {len(processed)} documents")
        return processed

    def get_document_stats(self, documents: List[Document]) -> Dict[str, object]:
        """
        Compute statistics about a document collection.

        Args:
            documents: Documents to analyze.

        Returns:
            Statistics dictionary with counts and file type breakdown.
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_characters": 0,
                "avg_length": 0,
                "file_types": {},
            }

        total_chars = sum(len(doc.text) for doc in documents)
        file_types: Dict[str, int] = {}

        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            ext = Path(file_name).suffix.lower().lstrip('.')
            file_types[ext] = file_types.get(ext, 0) + 1

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_length": total_chars // len(documents),
            "file_types": file_types,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean document text content.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text with normalized whitespace.
        """
        # Collapse whitespace
        text = ' '.join(text.split())
        # Remove null bytes
        text = text.replace('\x00', '')
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text

    def validate_documents(
        self,
        documents: List[Document],
    ) -> Tuple[List[Document], List[str]]:
        """
        Validate documents and separate valid from invalid.

        Args:
            documents: Documents to validate.

        Returns:
            Tuple of (valid_documents, error_messages).
        """
        valid_docs: List[Document] = []
        errors: List[str] = []

        for i, doc in enumerate(documents):
            try:
                if not doc.text or len(doc.text.strip()) < 10:
                    errors.append(f"Document {i}: Text too short or empty")
                    continue

                if not doc.metadata:
                    errors.append(f"Document {i}: Missing metadata")
                    continue

                valid_docs.append(doc)

            except Exception as e:
                errors.append(f"Document {i}: Validation error - {e}")

        if errors:
            logger.warning(f"Document validation found {len(errors)} issues")

        return valid_docs, errors