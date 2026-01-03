"""
Document processing pipeline.
Handles loading and preprocessing of documents.
"""

import logging
from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import TextNode
from config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents for indexing."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize processor.

        Args:
            data_dir: Directory containing documents
        """
        self.data_dir = data_dir or settings.DATA_DIR
        self.allowed_extensions = settings.get_allowed_extensions()

    def load_documents(self, file_paths: Optional[List[Path]] = None) -> List[Document]:
        """
        Load documents from data directory.

        Args:
            file_paths: Specific files to load. If None, loads all files.

        Returns:
            List of Document objects
        """
        try:
            if file_paths:
                # Load specific files
                documents = []
                for file_path in file_paths:
                    if file_path.exists():
                        reader = SimpleDirectoryReader(
                            input_files=[str(file_path)]
                        )
                        docs = reader.load_data()
                        documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} document(s) from {file_path.name}")
            else:
                # Load all files from directory
                if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
                    logger.warning("No documents found in data directory")
                    return []

                reader = SimpleDirectoryReader(
                    str(self.data_dir),
                    recursive=False,
                    filename_as_id=True
                )
                documents = reader.load_data()
                logger.info(f"Loaded {len(documents)} documents from {self.data_dir}")

            return documents

        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents before indexing.

        Args:
            documents: List of documents to preprocess

        Returns:
            List of preprocessed documents
        """
        processed = []

        for doc in documents:
            try:
                # Clean text
                text = self._clean_text(doc.text)

                # Skip if text is too short
                if len(text.strip()) < 10:
                    logger.warning(f"Skipping document with insufficient text")
                    continue

                # Create new document with cleaned text
                processed_doc = Document(
                    text=text,
                    metadata={
                        **doc.metadata,
                        "processed": True
                    }
                )

                processed.append(processed_doc)

            except Exception as e:
                logger.error(f"Failed to preprocess document: {e}")
                continue

        logger.info(f"Preprocessed {len(processed)} documents")
        return processed

    def get_document_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about documents.

        Args:
            documents: List of documents

        Returns:
            Statistics dictionary
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_characters": 0,
                "avg_length": 0,
                "file_types": {}
            }

        total_chars = sum(len(doc.text) for doc in documents)
        file_types = {}

        for doc in documents:
            file_name = doc.metadata.get("file_name", "unknown")
            ext = Path(file_name).suffix.lower().lstrip('.')
            file_types[ext] = file_types.get(ext, 0) + 1

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "avg_length": total_chars // len(documents),
            "file_types": file_types
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean document text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove null bytes
        text = text.replace('\x00', '')

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        return text

    def validate_documents(self, documents: List[Document]) -> tuple[List[Document], List[str]]:
        """
        Validate documents and return valid ones with error messages.

        Args:
            documents: Documents to validate

        Returns:
            Tuple of (valid_documents, error_messages)
        """
        valid_docs = []
        errors = []

        for i, doc in enumerate(documents):
            try:
                # Check if text exists
                if not doc.text or len(doc.text.strip()) < 10:
                    errors.append(f"Document {i}: Text too short or empty")
                    continue

                # Check metadata
                if not doc.metadata:
                    errors.append(f"Document {i}: Missing metadata")
                    continue

                valid_docs.append(doc)

            except Exception as e:
                errors.append(f"Document {i}: Validation error - {str(e)}")

        if errors:
            logger.warning(f"Document validation found {len(errors)} issues")

        return valid_docs, errors